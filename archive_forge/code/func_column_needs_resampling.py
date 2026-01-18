import numpy as np
from scipy.sparse.linalg import aslinearoperator
def column_needs_resampling(i, X, Y=None):
    n, t = X.shape
    v = X[:, i]
    if any((vectors_are_parallel(v, X[:, j]) for j in range(i))):
        return True
    if Y is not None:
        if any((vectors_are_parallel(v, w) for w in Y.T)):
            return True
    return False