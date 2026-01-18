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
def get_sym_eq_kpoints(self, kpoint, cartesian=False, tol: float=0.01):
    """Returns a list of unique symmetrically equivalent k-points.

        Args:
            kpoint (1x3 array): coordinate of the k-point
            cartesian (bool): kpoint is in Cartesian or fractional coordinates
            tol (float): tolerance below which coordinates are considered equal

        Returns:
            list[1x3 array] | None: if structure is not available returns None
        """
    if not self.structure:
        return None
    sg = SpacegroupAnalyzer(self.structure)
    symm_ops = sg.get_point_group_operations(cartesian=cartesian)
    points = np.dot(kpoint, [m.rotation_matrix for m in symm_ops])
    rm_list = []
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            if np.allclose(pbc_diff(points[i], points[j]), [0, 0, 0], tol):
                rm_list.append(i)
                break
    return np.delete(points, rm_list, axis=0)