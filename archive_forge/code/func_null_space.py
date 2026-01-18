from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def null_space(x):
    aux1 = Vt.dot(x)
    aux2 = 1 / s * aux1
    v = U.dot(aux2)
    z = x - A.T.dot(v)
    k = 0
    while orthogonality(A, z) > orth_tol:
        if k >= max_refin:
            break
        aux1 = Vt.dot(z)
        aux2 = 1 / s * aux1
        v = U.dot(aux2)
        z = z - A.T.dot(v)
        k += 1
    return z