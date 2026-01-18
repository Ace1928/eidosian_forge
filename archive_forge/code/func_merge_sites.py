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
def merge_sites(self, tol: float=0.01, mode: Literal['sum', 'delete', 'average']='sum') -> Self:
    """Merges sites (adding occupancies) within tol of each other.
        Removes site properties.

        Args:
            tol (float): Tolerance for distance to merge sites.
            mode ('sum' | 'delete' | 'average'): "delete" means duplicate sites are
                deleted. "sum" means the occupancies are summed for the sites.
                "average" means that the site is deleted but the properties are averaged
                Only first letter is considered.

        Returns:
            Structure: self with merged sites.
        """
    dist_mat = self.distance_matrix
    np.fill_diagonal(dist_mat, 0)
    clusters = fcluster(linkage(squareform((dist_mat + dist_mat.T) / 2)), tol, 'distance')
    sites = []
    for c in np.unique(clusters):
        inds = np.where(clusters == c)[0]
        species = self[inds[0]].species
        coords = self[inds[0]].frac_coords
        props = self[inds[0]].properties
        for n, i in enumerate(inds[1:]):
            sp = self[i].species
            if mode.lower()[0] == 's':
                species += sp
            offset = self[i].frac_coords - coords
            coords = coords + ((offset - np.round(offset)) / (n + 2)).astype(coords.dtype)
            for key in props:
                if props[key] is not None and self[i].properties[key] != props[key]:
                    if mode.lower()[0] == 'a' and isinstance(props[key], float):
                        props[key] = props[key] * (n + 1) / (n + 2) + self[i].properties[key] / (n + 2)
                    else:
                        props[key] = None
                        warnings.warn(f'Sites with different site property {key} are merged. So property is set to none')
        sites.append(PeriodicSite(species, coords, self.lattice, properties=props))
    self._sites = sites
    return self