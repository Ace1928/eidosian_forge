import numpy as np
from scipy.sparse.linalg import aslinearoperator
def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    for v in X.T:
        if not any((vectors_are_parallel(v, w) for w in Y.T)):
            return False
    return True