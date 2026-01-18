from scipy import sparse
import numbers
import numpy as np
def set_diagonal(X, diag):
    return if_sparse(sparse_set_diagonal, dense_set_diagonal, X, diag=diag)