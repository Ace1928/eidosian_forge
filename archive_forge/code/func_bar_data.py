from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
@staticmethod
def bar_data(A_psd, c_psd, K):
    n = A_psd.shape[0]
    c_bar_data, A_bar_data = ([], [])
    idx = 0
    for j, dim in enumerate(K[a2d.PSD]):
        vec_len = dim * (dim + 1) // 2
        A_block = A_psd[:, idx:idx + vec_len]
        for i in range(n):
            A_row = A_block[i, :]
            if A_row.nnz == 0:
                continue
            A_row_coo = A_row.tocoo()
            rows, cols, vals = vectorized_lower_tri_to_triples(A_row_coo, dim)
            A_bar_data.append((i, j, (rows, cols, vals)))
        c_block = c_psd[idx:idx + vec_len]
        rows, cols, vals = vectorized_lower_tri_to_triples(c_block, dim)
        c_bar_data.append((j, (rows, cols, vals)))
        idx += vec_len
    return (A_bar_data, c_bar_data)