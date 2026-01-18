from scipy import sparse
import numbers
import numpy as np
def if_sparse(sparse_func, dense_func, *args, **kwargs):
    if sparse.issparse(args[0]):
        for arg in args[1:]:
            assert sparse.issparse(arg)
        return sparse_func(*args, **kwargs)
    else:
        return dense_func(*args, **kwargs)