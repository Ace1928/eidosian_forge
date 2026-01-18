from scipy import sparse
import numbers
import numpy as np
def nonzero_discrete(X, values):
    if isinstance(values, numbers.Number):
        values = [values]
    if 0 not in values:
        values.append(0)
    return if_sparse(sparse_nonzero_discrete, dense_nonzero_discrete, X, values=values)