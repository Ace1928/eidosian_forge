import numpy as np
from scipy import sparse
from pygsp import utils
def lanczos(A, order, x):
    """
    TODO short description

    Parameters
    ----------
    A: ndarray

    Returns
    -------
    """
    try:
        N, M = np.shape(x)
    except ValueError:
        N = np.shape(x)[0]
        M = 1
        x = x[:, np.newaxis]
    q = np.divide(x, np.kron(np.ones((N, 1)), np.linalg.norm(x, axis=0)))
    hiv = np.arange(0, order * M, order)
    V = np.zeros((N, M * order))
    V[:, hiv] = q
    H = np.zeros((order + 1, M * order))
    r = np.dot(A, q)
    H[0, hiv] = np.sum(q * r, axis=0)
    r -= np.kron(np.ones((N, 1)), H[0, hiv]) * q
    H[1, hiv] = np.linalg.norm(r, axis=0)
    orth = np.zeros(order)
    orth[0] = np.linalg.norm(np.dot(V.T, V) - M)
    for k in range(1, order):
        if np.sum(np.abs(H[k, hiv + k - 1])) <= np.spacing(1):
            H = H[:k - 1, _sum_ind(np.arange(k), hiv)]
            V = V[:, _sum_ind(np.arange(k), hiv)]
            orth = orth[:k]
            return (V, H, orth)
        H[k - 1, hiv + k] = H[k, hiv + k - 1]
        v = q
        q = r / np.tile(H[k - 1, k + hiv], (N, 1))
        V[:, k + hiv] = q
        r = np.dot(A, q)
        r -= np.tile(H[k - 1, k + hiv], (N, 1)) * v
        H[k, k + hiv] = np.sum(np.multiply(q, r), axis=0)
        r -= np.tile(H[k, k + hiv], (N, 1)) * q
        r -= np.dot(V, np.dot(V.T, r))
        H[k + 1, k + hiv] = np.linalg.norm(r, axis=0)
        orth[k] = np.linalg.norm(np.dot(V.T, V) - M)
    H = H[np.ix_(np.arange(order), np.arange(order))]
    return (V, H, orth)