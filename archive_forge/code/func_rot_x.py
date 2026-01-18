import numpy as np
from matplotlib import _api
@_api.deprecated('3.8')
def rot_x(V, alpha):
    cosa, sina = (np.cos(alpha), np.sin(alpha))
    M1 = np.array([[1, 0, 0, 0], [0, cosa, -sina, 0], [0, sina, cosa, 0], [0, 0, 0, 1]])
    return np.dot(M1, V)