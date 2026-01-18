from cupy.random import _generator
from cupy import _util
def logistic(loc=0.0, scale=1.0, size=None, dtype=float):
    """Logistic distribution.

    Returns an array of samples drawn from the logistic distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2}.

    Args:
        loc (float): The location of the mode :math:`\\mu`.
        scale (float): The scale parameter :math:`s`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the logistic distribution.

    .. seealso::
        :func:`numpy.random.logistic`
    """
    rs = _generator.get_random_state()
    return rs.logistic(loc, scale, size, dtype)