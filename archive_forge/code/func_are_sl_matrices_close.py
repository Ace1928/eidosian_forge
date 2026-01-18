from snappy.SnapPy import matrix
from ..upper_halfspace.ideal_point import Infinity
def are_sl_matrices_close(m1, m2, epsilon=1e-05):
    """
    Compute whether two matrices are the same up to given epsilon.
    """
    for i in range(2):
        for j in range(2):
            if abs(m1[i, j] - m2[i, j]) > epsilon:
                return False
    return True