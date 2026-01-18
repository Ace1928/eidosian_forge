from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def least_squares(x):
    aux1 = Vt.dot(x)
    aux2 = 1 / s * aux1
    z = U.dot(aux2)
    return z