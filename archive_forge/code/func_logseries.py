from cupy.random import _generator
from cupy import _util
def logseries(p, size=None, dtype=int):
    """Log series distribution.

    Returns an array of samples drawn from the log series distribution. Its
    probability mass function is defined as

    .. math::
       f(x) = \\frac{-p^x}{x\\ln(1-p)}.

    Args:
        p (float): Parameter of the log series distribution :math:`p`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the log series distribution.

    .. seealso:: :func:`numpy.random.logseries`

    """
    rs = _generator.get_random_state()
    return rs.logseries(p, size=size, dtype=dtype)