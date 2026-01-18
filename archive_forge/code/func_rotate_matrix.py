import numpy as np
def rotate_matrix(h, u):
    return np.dot(u.T.conj(), np.dot(h, u))