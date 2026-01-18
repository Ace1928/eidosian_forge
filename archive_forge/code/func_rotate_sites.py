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
def rotate_sites(self, indices: Sequence[int] | None=None, theta: float=0.0, axis: ArrayLike | None=None, anchor: ArrayLike | None=None) -> Molecule:
    """Rotate specific sites by some angle around vector at anchor.

        Args:
            indices (list): List of site indices on which to perform the
                translation.
            theta (float): Angle in radians
            axis (3x1 array): Rotation axis vector.
            anchor (3x1 array): Point of rotation.

        Returns:
            Molecule: self with rotated sites.
        """
    if indices is None:
        indices = range(len(self))
    if axis is None:
        axis = [0, 0, 1]
    if anchor is None:
        anchor = [0, 0, 0]
    anchor = np.array(anchor)
    axis = np.array(axis)
    theta %= 2 * np.pi
    rm = expm(cross(eye(3), axis / norm(axis)) * theta)
    for idx in indices:
        site = self[idx]
        coords = (np.dot(rm, (site.coords - anchor).T).T + anchor).ravel()
        new_site = Site(site.species, coords, properties=site.properties, label=site.label)
        self[idx] = new_site
    return self