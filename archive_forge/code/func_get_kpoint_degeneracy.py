from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
def get_kpoint_degeneracy(self, kpoint, cartesian=False, tol: float=0.01):
    """Returns degeneracy of a given k-point based on structure symmetry.

        Args:
            kpoint (1x3 array): coordinate of the k-point
            cartesian (bool): kpoint is in Cartesian or fractional coordinates
            tol (float): tolerance below which coordinates are considered equal.

        Returns:
            int | None: degeneracy or None if structure is not available
        """
    all_kpts = self.get_sym_eq_kpoints(kpoint, cartesian, tol=tol)
    if all_kpts is not None:
        return len(all_kpts)
    return None