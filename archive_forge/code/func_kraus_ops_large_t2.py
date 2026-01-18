import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
def kraus_ops_large_t2():
    e0 = p_reset * pe
    v0 = np.array([[0, 0], [1, 0]])
    K0 = np.sqrt(e0 + np.eps) * v0
    e1 = -p_reset * pe + p_reset
    v1 = np.array([[0, 1], [0, 0]])
    K1 = np.sqrt(e1 + np.eps) * v1
    base = sum((4 * eT2 ** 2, 4 * p_reset ** 2 * pe ** 2, -4 * p_reset ** 2 * pe, p_reset ** 2, np.eps))
    common_term = np.sqrt(base)
    e2 = 1 - p_reset / 2 - common_term / 2
    term2 = 2 * eT2 / (2 * p_reset * pe - p_reset - common_term)
    v2 = (term2 * np.array([[1, 0], [0, 0]]) + np.array([[0, 0], [0, 1]])) / np.sqrt(term2 ** 2 + 1)
    K2 = np.sqrt(e2 + np.eps) * v2
    term3 = 2 * eT2 / (2 * p_reset * pe - p_reset + common_term)
    e3 = 1 - p_reset / 2 + common_term / 2
    v3 = (term3 * np.array([[1, 0], [0, 0]]) + np.array([[0, 0], [0, 1]])) / np.sqrt(term3 ** 2 + 1)
    K3 = np.sqrt(e3 + np.eps) * v3
    K4 = np.cast_like(np.zeros((2, 2)), K1)
    K5 = np.cast_like(np.zeros((2, 2)), K1)
    return [K0, K1, K2, K3, K4, K5]