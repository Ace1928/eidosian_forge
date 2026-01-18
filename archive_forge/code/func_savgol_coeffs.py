import math
import cupy
from cupy.linalg import lstsq
from cupyx.scipy.ndimage import convolve1d
from ._arraytools import axis_slice
def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None, use='conv'):
    """Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : 1-D ndarray
        The filter coefficients.

    See Also
    --------
    scipy.signal.savgol_coeffs
    savgol_filter


    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.
    Jianwen Luo, Kui Ying, and Jing Bai. 2005. Savitzky-Golay smoothing and
    differentiation filter for even number data. Signal Process.
    85, 7 (July 2005), 1429-1434.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import savgol_coeffs
    >>> savgol_coeffs(5, 2)
    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])
    >>> savgol_coeffs(5, 2, deriv=1)
    array([ 2.00000000e-01,  1.00000000e-01,  2.07548111e-16, -1.00000000e-01,
           -2.00000000e-01])

    Note that use='dot' simply reverses the coefficients.

    >>> savgol_coeffs(5, 2, pos=3)
    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])
    >>> savgol_coeffs(5, 2, pos=3, use='dot')
    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])
    >>> savgol_coeffs(4, 2, pos=3, deriv=1, use='dot')
    array([0.45,  -0.85,  -0.65,  1.05])

    `x` contains data from the parabola x = t**2, sampled at
    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
    derivative at the last position.  When dotted with `x` the result should
    be 6.

    >>> x = np.array([1, 0, 1, 4, 9])
    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
    >>> c.dot(x)
    6.0
    """
    if polyorder >= window_length:
        raise ValueError('polyorder must be less than window_length.')
    halflen, rem = divmod(window_length, 2)
    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen
    if not 0 <= pos < window_length:
        raise ValueError('pos must be nonnegative and less than window_length.')
    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")
    if deriv > polyorder:
        coeffs = cupy.zeros(window_length)
        return coeffs
    x = cupy.arange(-pos, window_length - pos, dtype=float)
    if use == 'conv':
        x = x[::-1]
    order = cupy.arange(polyorder + 1).reshape(-1, 1)
    A = x ** order
    y = cupy.zeros(polyorder + 1)
    y[deriv] = float_factorial(deriv) / delta ** deriv
    coeffs, _, _, _ = lstsq(A, y, rcond=None)
    return coeffs