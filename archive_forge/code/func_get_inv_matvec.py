import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import isdense, is_pydata_spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock
from . import _arpack
def get_inv_matvec(M, hermitian=False, tol=0):
    if isdense(M):
        return LuInv(M).matvec
    elif issparse(M) or is_pydata_spmatrix(M):
        M = _fast_spmatrix_to_csc(M, hermitian=hermitian)
        return SpLuInv(M).matvec
    else:
        return IterInv(M, tol=tol).matvec