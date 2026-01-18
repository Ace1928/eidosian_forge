from scipy import sparse
import numbers
import numpy as np
def elementwise_maximum(X, Y):
    return if_sparse(sparse_maximum, np.maximum, X, Y)