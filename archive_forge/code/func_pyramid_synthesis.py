import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def pyramid_synthesis(Gs, cap, pe, order=30, **kwargs):
    """Synthesize a signal from its pyramid coefficients.

    Parameters
    ----------
    Gs : Array of Graphs
        A multiresolution sequence of graph structures.
    cap : ndarray
        Coarsest approximation of the original signal.
    pe : ndarray
        Prediction error at each level.
    use_exact : bool
        To use exact graph spectral filtering instead of the Chebyshev approximation.
    order : int
        Degree of the Chebyshev approximation (default=30).
    least_squares : bool
        To use the least squares synthesis (default=False).
    h_filters : ndarray
        The filters used in the analysis operator.
        These are required for least squares synthesis, but not for the direct synthesis method.
    use_landweber : bool
        To use the Landweber iteration approximation in the least squares synthesis.
    reg_eps : float
        Interpolation parameter.
    landweber_its : int
        Number of iterations in the Landweber approximation for least squares synthesis.
    landweber_tau : float
        Parameter for the Landweber iteration.

    Returns
    -------
    reconstruction : ndarray
        The reconstructed signal.
    ca : ndarray
        Coarse approximations at each level

    """
    least_squares = bool(kwargs.pop('least_squares', False))
    def_ul = Gs[0].N > 3000 or not hasattr(Gs[0], '_e') or (not hasattr(Gs[0], '_U'))
    use_landweber = bool(kwargs.pop('use_landweber', def_ul))
    reg_eps = float(kwargs.get('reg_eps', 0.005))
    if least_squares and 'h_filters' not in kwargs:
        ValueError('h-filters not provided.')
    levels = len(Gs) - 1
    if len(pe) != levels:
        ValueError('Gs and pe have different shapes.')
    ca = [cap]
    for i in range(levels):
        if not least_squares:
            s_pred = interpolate(Gs[levels - i - 1], ca[i], Gs[levels - i].mr['idx'], order=order, reg_eps=reg_eps, **kwargs)
            ca.append(s_pred + pe[levels - i - 1])
        else:
            ca.append(_pyramid_single_interpolation(Gs[levels - i - 1], ca[i], pe[levels - i - 1], h_filters[levels - i - 1], use_landweber=use_landweber, **kwargs))
    ca.reverse()
    reconstruction = ca[0]
    return (reconstruction, ca)