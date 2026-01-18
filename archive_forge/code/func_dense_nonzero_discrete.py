from scipy import sparse
import numbers
import numpy as np
def dense_nonzero_discrete(X, values):
    result = np.full_like(X, False, dtype=bool)
    for value in values:
        result = np.logical_or(result, X == value)
    return np.all(result)