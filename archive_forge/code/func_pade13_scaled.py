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
def pade13_scaled(self, s):
    b = (6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0)
    B = self.A * 2 ** (-s)
    B2 = self.A2 * 2 ** (-2 * s)
    B4 = self.A4 * 2 ** (-4 * s)
    B6 = self.A6 * 2 ** (-6 * s)
    U2 = _smart_matrix_product(B6, b[13] * B6 + b[11] * B4 + b[9] * B2, structure=self.structure)
    U = _smart_matrix_product(B, U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * self.ident, structure=self.structure)
    V2 = _smart_matrix_product(B6, b[12] * B6 + b[10] * B4 + b[8] * B2, structure=self.structure)
    V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * self.ident
    return (U, V)