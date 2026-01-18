from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
from warnings import warn
def normal_equation_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``NormalEquation`` approach.
    """
    factor = cholesky_AAt(A)

    def null_space(x):
        v = factor(A.dot(x))
        z = x - A.T.dot(v)
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            v = factor(A.dot(z))
            z = z - A.T.dot(v)
            k += 1
        return z

    def least_squares(x):
        return factor(A.dot(x))

    def row_space(x):
        return A.T.dot(factor(x))
    return (null_space, least_squares, row_space)