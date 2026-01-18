import numpy as np
from scipy import sparse
from pygsp import utils
def compute_jackson_cheby_coeff(filter_bounds, delta_lambda, m):
    """
    To compute the m+1 coefficients of the polynomial approximation of an ideal band-pass between a and b, between a range of values defined by lambda_min and lambda_max.

    Parameters
    ----------
    filter_bounds : list
        [a, b]
    delta_lambda : list
        [lambda_min, lambda_max]
    m : int

    Returns
    -------
    ch : ndarray
    jch : ndarray

    References
    ----------
    :cite:`tremblay2016compressive`

    """
    if delta_lambda[0] > filter_bounds[0] or delta_lambda[1] < filter_bounds[1]:
        _logger.error('Bounds of the filter are out of the lambda values')
        raise ()
    elif delta_lambda[0] > delta_lambda[1]:
        _logger.error('lambda_min is greater than lambda_max')
        raise ()
    a1 = (delta_lambda[1] - delta_lambda[0]) / 2
    a2 = (delta_lambda[1] + delta_lambda[0]) / 2
    filter_bounds[0] = (filter_bounds[0] - a2) / a1
    filter_bounds[1] = (filter_bounds[1] - a2) / a1
    ch = np.arange(float(m + 1))
    ch[0] = 2 / np.pi * (np.arccos(filter_bounds[0]) - np.arccos(filter_bounds[1]))
    for i in ch[1:]:
        ch[i] = 2 / (np.pi * i) * (np.sin(i * np.arccos(filter_bounds[0])) - np.sin(i * np.arccos(filter_bounds[1])))
    jch = np.arange(float(m + 1))
    alpha = np.pi / (m + 2)
    for i in jch:
        jch[i] = 1 / np.sin(alpha) * ((1 - i / (m + 2)) * np.sin(alpha) * np.cos(i * alpha) + 1 / (m + 2) * np.cos(alpha) * np.sin(i * alpha))
    jch = ch * jch
    return (ch, jch)