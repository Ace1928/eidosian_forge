from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def reduce_mat(mat, mag, r_matrix):
    """
        Reduce integer array mat's determinant mag times by linear combination
        of its row vectors, so that the new array after rotation (r_matrix) is
        still an integer array.

        Args:
            mat (3 by 3 array): input matrix
            mag (int): reduce times for the determinant
            r_matrix (3 by 3 array): rotation matrix

        Returns:
            the reduced integer array
        """
    max_j = abs(int(round(np.linalg.det(mat) / mag)))
    reduced = False
    for h in range(3):
        kk = h + 1 if h + 1 < 3 else abs(2 - h)
        ll = h + 2 if h + 2 < 3 else abs(1 - h)
        jj = np.arange(-max_j, max_j + 1)
        for j1, j2 in product(jj, repeat=2):
            temp = mat[h] + j1 * mat[kk] + j2 * mat[ll]
            if all((np.round(x, 5).is_integer() for x in list(temp / mag))):
                mat_copy = mat.copy()
                mat_copy[h] = np.array([int(round(ele / mag)) for ele in temp])
                new_mat = np.dot(mat_copy, np.linalg.inv(r_matrix.T))
                if all((np.round(x, 5).is_integer() for x in list(np.ravel(new_mat)))):
                    reduced = True
                    mat[h] = np.array([int(round(ele / mag)) for ele in temp])
                    break
        if reduced:
            break
    if not reduced:
        warnings.warn('Matrix reduction not performed, may lead to non-primitive GB cell.')
    return mat