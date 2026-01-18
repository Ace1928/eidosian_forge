import numpy as np
from scipy.sparse.linalg import aslinearoperator
def elementary_vector(n, i):
    v = np.zeros(n, dtype=float)
    v[i] = 1
    return v