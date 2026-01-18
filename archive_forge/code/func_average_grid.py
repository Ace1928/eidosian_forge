import warnings
import numpy as np
from scipy import interpolate, stats
def average_grid(values, coords=None, _method='slicing'):
    """Compute average for each cell in grid using endpoints

    Parameters
    ----------
    values : array_like
        Values on a grid that will average over corner points of each cell.
    coords : None or list of array_like
        Grid coordinates for each axis use to compute volumne of cell.
        If None, then averaged values are not rescaled.
    _method : {"slicing", "convolve"}
        Grid averaging is implemented using numpy "slicing" or using
        scipy.signal "convolve".

    Returns
    -------
    Grid with averaged cell values.
    """
    k_dim = values.ndim
    if _method == 'slicing':
        p = values.copy()
        for d in range(k_dim):
            sl1 = [slice(None, None, None)] * k_dim
            sl2 = [slice(None, None, None)] * k_dim
            sl1[d] = slice(None, -1, None)
            sl2[d] = slice(1, None, None)
            sl1 = tuple(sl1)
            sl2 = tuple(sl2)
            p = (p[sl1] + p[sl2]) / 2
    elif _method == 'convolve':
        from scipy import signal
        p = signal.convolve(values, 0.5 ** k_dim * np.ones([2] * k_dim), mode='valid')
    if coords is not None:
        dx = np.array(1)
        for d in range(k_dim):
            dx = dx[..., None] * np.diff(coords[d])
        p = p * dx
    return p