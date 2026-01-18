import numpy as np
from numpy.linalg import norm, lstsq
from scipy.optimize import OptimizeResult
from .common import print_header_linear, print_iteration_linear
def compute_kkt_optimality(g, on_bound):
    """Compute the maximum violation of KKT conditions."""
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = np.abs(g[free_set])
    return np.max(g_kkt)