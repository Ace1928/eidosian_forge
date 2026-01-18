import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like
def distance_indicators(x, epsilon=None, distance=1.5):
    """
    Calculate all pairwise threshold distance indicators for a time series

    Parameters
    ----------
    x : 1d array
        observations of time series for which heaviside distance indicators
        are calculated
    epsilon : scalar, optional
        the threshold distance to use in calculating the heaviside indicators
    distance : scalar, optional
        if epsilon is omitted, specifies the distance multiplier to use when
        computing it

    Returns
    -------
    indicators : 2d array
        matrix of distance threshold indicators

    Notes
    -----
    Since this can be a very large matrix, use np.int8 to save some space.
    """
    x = array_like(x, 'x')
    if epsilon is not None and epsilon <= 0:
        raise ValueError('Threshold distance must be positive if specified. Got epsilon of %f' % epsilon)
    if distance <= 0:
        raise ValueError('Threshold distance must be positive. Got distance multiplier %f' % distance)
    if epsilon is None:
        epsilon = distance * x.std(ddof=1)
    return np.abs(x[:, None] - x) < epsilon