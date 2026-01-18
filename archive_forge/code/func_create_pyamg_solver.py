import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def create_pyamg_solver(P, max_levels=15):
    return smoothed_aggregation_solver(P, B=None, strength=('symmetric', {'theta': 0.0}), smooth=('jacobi', {'filter': True, 'weighting': 'local'}), improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})] + [None] * (max_levels - 1), aggregate='standard', presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}), postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}), max_levels=max_levels, max_coarse=300, coarse_solver='pinv')