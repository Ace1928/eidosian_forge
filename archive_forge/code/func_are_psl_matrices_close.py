from snappy.SnapPy import matrix
from ..upper_halfspace.ideal_point import Infinity
def are_psl_matrices_close(m1, m2, epsilon=1e-05):
    """
    Compute whether two matrices are the same up to given epsilon
    and multiplying by -Identity.
    """
    return are_sl_matrices_close(m1, m2, epsilon) or are_sl_matrices_close(m1, -m2, epsilon)