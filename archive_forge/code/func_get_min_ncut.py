import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
def get_min_ncut(ev, d, w, num_cuts):
    """Threshold an eigenvector evenly, to determine minimum ncut.

    Parameters
    ----------
    ev : array
        The eigenvector to threshold.
    d : ndarray
        The diagonal matrix of the graph.
    w : ndarray
        The weight matrix of the graph.
    num_cuts : int
        The number of evenly spaced thresholds to check for.

    Returns
    -------
    mask : array
        The array of booleans which denotes the bi-partition.
    mcut : float
        The value of the minimum ncut.
    """
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return (min_mask, mcut)
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = _ncut.ncut_cost(mask, d, w)
        if cost < mcut:
            min_mask = mask
            mcut = cost
    return (min_mask, mcut)