import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def create_bandmat(self):
    """Create the full matrix `self.fullmat` and
           the corresponding band matrix `self.bandmat`."""
    N = 10
    self.KL = 2
    self.KU = 2
    self.sym_mat = diag(full(N, 1.0)) + diag(full(N - 1, -1.0), -1) + diag(full(N - 1, -1.0), 1) + diag(full(N - 2, -2.0), -2) + diag(full(N - 2, -2.0), 2)
    self.herm_mat = diag(full(N, -1.0)) + 1j * diag(full(N - 1, 1.0), -1) - 1j * diag(full(N - 1, 1.0), 1) + diag(full(N - 2, -2.0), -2) + diag(full(N - 2, -2.0), 2)
    self.real_mat = diag(full(N, 1.0)) + diag(full(N - 1, -1.0), -1) + diag(full(N - 1, -3.0), 1) + diag(full(N - 2, 2.0), -2) + diag(full(N - 2, -2.0), 2)
    self.comp_mat = 1j * diag(full(N, 1.0)) + diag(full(N - 1, -1.0), -1) + 1j * diag(full(N - 1, -3.0), 1) + diag(full(N - 2, 2.0), -2) + diag(full(N - 2, -2.0), 2)
    ew, ev = linalg.eig(self.sym_mat)
    ew = ew.real
    args = argsort(ew)
    self.w_sym_lin = ew[args]
    self.evec_sym_lin = ev[:, args]
    ew, ev = linalg.eig(self.herm_mat)
    ew = ew.real
    args = argsort(ew)
    self.w_herm_lin = ew[args]
    self.evec_herm_lin = ev[:, args]
    LDAB = self.KU + 1
    self.bandmat_sym = zeros((LDAB, N), dtype=float)
    self.bandmat_herm = zeros((LDAB, N), dtype=complex)
    for i in range(LDAB):
        self.bandmat_sym[LDAB - i - 1, i:N] = diag(self.sym_mat, i)
        self.bandmat_herm[LDAB - i - 1, i:N] = diag(self.herm_mat, i)
    LDAB = 2 * self.KL + self.KU + 1
    self.bandmat_real = zeros((LDAB, N), dtype=float)
    self.bandmat_real[2 * self.KL, :] = diag(self.real_mat)
    for i in range(self.KL):
        self.bandmat_real[2 * self.KL - 1 - i, i + 1:N] = diag(self.real_mat, i + 1)
        self.bandmat_real[2 * self.KL + 1 + i, 0:N - 1 - i] = diag(self.real_mat, -i - 1)
    self.bandmat_comp = zeros((LDAB, N), dtype=complex)
    self.bandmat_comp[2 * self.KL, :] = diag(self.comp_mat)
    for i in range(self.KL):
        self.bandmat_comp[2 * self.KL - 1 - i, i + 1:N] = diag(self.comp_mat, i + 1)
        self.bandmat_comp[2 * self.KL + 1 + i, 0:N - 1 - i] = diag(self.comp_mat, -i - 1)
    self.b = 1.0 * arange(N)
    self.bc = self.b * (1 + 1j)