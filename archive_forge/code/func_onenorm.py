from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def onenorm(self):
    """
        Compute the exact 1-norm.
        """
    if self._A_1_norm is None:
        self._A_1_norm = _exact_1_norm(self._A)
    return self._scale * self._A_1_norm