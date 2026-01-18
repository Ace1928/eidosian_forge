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
def one_dim_to_ndim(self, csc_P, N):
    """
        Expand an N x N precon matrix to self.dim*N x self.dim * N

        Args:
            csc_P (sparse matrix): N x N sparse matrix, in CSC format
        """
    start_time = time.time()
    if self.dim == 1:
        P = csc_P
    elif self.array_convention == 'F':
        csc_P = csc_P.tocsr()
        P = csc_P
        for i in range(self.dim - 1):
            P = sparse.block_diag((P, csc_P)).tocsr()
    else:
        csc_P = csc_P.tocoo()
        i = csc_P.row * self.dim
        j = csc_P.col * self.dim
        z = csc_P.data
        I = np.hstack([i + d for d in range(self.dim)])
        J = np.hstack([j + d for d in range(self.dim)])
        Z = np.hstack([z for d in range(self.dim)])
        P = sparse.csc_matrix((Z, (I, J)), shape=(self.dim * N, self.dim * N))
        P = P.tocsr()
    self.logfile.write('--- N-dim precon created in %s s ---\n' % (time.time() - start_time))
    return P