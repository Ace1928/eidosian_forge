from cupy.random import _generator
from cupy import _util
def standard_gamma(shape, size=None, dtype=float):
    """Standard gamma distribution.

    Returns an array of samples drawn from the standard gamma distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{1}{\\Gamma(k)}x^{k-1}e^{-x}.

    Args:
        shape (array): Parameter of the gamma distribution :math:`k`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard gamma distribution.

    .. seealso::
        :func:`numpy.random.standard_gamma`
    """
    rs = _generator.get_random_state()
    return rs.standard_gamma(shape, size, dtype)