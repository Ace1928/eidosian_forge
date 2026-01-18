import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
@property
def d4_loose(self):
    if self.use_exact_onenorm:
        return self.d4_tight
    if self._d4_exact is not None:
        return self._d4_exact
    else:
        if self._d4_approx is None:
            self._d4_approx = _onenormest_matrix_power(self.A2, 2, structure=self.structure) ** (1 / 4.0)
        return self._d4_approx