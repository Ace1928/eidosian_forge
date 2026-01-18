from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def row_space(x):
    aux1 = U.T.dot(x)
    aux2 = 1 / s * aux1
    z = Vt.T.dot(aux2)
    return z