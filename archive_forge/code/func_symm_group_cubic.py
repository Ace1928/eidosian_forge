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
def symm_group_cubic(mat):
    """
    Obtain cubic symmetric equivalents of the list of vectors.

    Args:
        mat (n by 3 array/matrix): lattice matrix


    Returns:
        cubic symmetric equivalents of the list of vectors.
    """
    sym_group = np.zeros([24, 3, 3])
    sym_group[0, :] = np.eye(3)
    sym_group[1, :] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    sym_group[2, :] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    sym_group[3, :] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    sym_group[4, :] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    sym_group[5, :] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    sym_group[6, :] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    sym_group[7, :] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    sym_group[8, :] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    sym_group[9, :] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    sym_group[10, :] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    sym_group[11, :] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    sym_group[12, :] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    sym_group[13, :] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    sym_group[14, :] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    sym_group[15, :] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    sym_group[16, :] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    sym_group[17, :] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    sym_group[18, :] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    sym_group[19, :] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    sym_group[20, :] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    sym_group[21, :] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    sym_group[22, :] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    sym_group[23, :] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    mat = np.atleast_2d(mat)
    all_vectors = []
    for sym in sym_group:
        for vec in mat:
            all_vectors.append(np.dot(sym, vec))
    return np.unique(np.array(all_vectors), axis=0)