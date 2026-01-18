from cupy.random import _generator
from cupy import _util
def standard_exponential(size=None, dtype=float):
    """Standard exponential distribution.

    Returns an array of samples drawn from the standard exponential
    distribution. Its probability density function is defined as

      .. math::
         f(x) = e^{-x}.

    Args:
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the standard exponential distribution.

    .. seealso:: :func:`numpy.random.standard_exponential`
    """
    rs = _generator.get_random_state()
    return rs.standard_exponential(size, dtype)