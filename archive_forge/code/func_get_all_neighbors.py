from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
def get_all_neighbors(self, r: float, include_index: bool=False, include_image: bool=False, sites: Sequence[PeriodicSite] | None=None, numerical_tol: float=1e-08) -> list[list[PeriodicNeighbor]]:
    """Get neighbors for each atom in the unit cell, out to a distance r
        Returns a list of list of neighbors for each site in structure.
        Use this method if you are planning on looping over all sites in the
        crystal. If you only want neighbors for a particular site, use the
        method get_neighbors as it may not have to build such a large supercell
        However if you are looping over all sites in the crystal, this method
        is more efficient since it only performs one pass over a large enough
        supercell to contain all possible atoms out to a distance r.
        The return type is a [(site, dist) ...] since most of the time,
        subsequent processing requires the distance.

        A note about periodic images: Before computing the neighbors, this
        operation translates all atoms to within the unit cell (having
        fractional coordinates within [0,1)). This means that the "image" of a
        site does not correspond to how much it has been translates from its
        current position, but which image of the unit cell it resides.

        Args:
            r (float): Radius of sphere.
            include_index (bool): Deprecated. Now, the non-supercell site index
                is always included in the returned data.
            include_image (bool): Deprecated. Now the supercell image
                is always included in the returned data.
            sites (list of Sites or None): sites for getting all neighbors,
                default is None, which means neighbors will be obtained for all
                sites. This is useful in the situation where you are interested
                only in one subspecies type, and makes it a lot faster.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.

        Returns:
            [[pymatgen.core.structure.PeriodicNeighbor], ..]
        """
    if sites is None:
        sites = self.sites
    center_indices, points_indices, images, distances = self.get_neighbor_list(r=r, sites=sites, numerical_tol=numerical_tol)
    if len(points_indices) < 1:
        return [[]] * len(sites)
    f_coords = self.frac_coords[points_indices] + images
    neighbor_dict: dict[int, list] = collections.defaultdict(list)
    lattice = self.lattice
    atol = Site.position_atol
    all_sites = self.sites
    for cindex, pindex, image, f_coord, d in zip(center_indices, points_indices, images, f_coords, distances):
        psite = all_sites[pindex]
        csite = sites[cindex]
        if d > numerical_tol or psite.species != csite.species or (not np.allclose(psite.coords, csite.coords, atol=atol)) or (psite.properties != csite.properties):
            neighbor_dict[cindex].append(PeriodicNeighbor(species=psite.species, coords=f_coord, lattice=lattice, properties=psite.properties, nn_distance=d, index=pindex, image=tuple(image), label=psite.label))
    neighbors: list[list[PeriodicNeighbor]] = []
    for i in range(len(sites)):
        neighbors.append(neighbor_dict[i])
    return neighbors