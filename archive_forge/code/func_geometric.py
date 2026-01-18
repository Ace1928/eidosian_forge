from cupy.random import _generator
from cupy import _util
def geometric(p, size=None, dtype=int):
    """Geometric distribution.

    Returns an array of samples drawn from the geometric distribution. Its
    probability mass function is defined as

    .. math::
        f(x) = p(1-p)^{k-1}.

    Args:
        p (float): Success probability of the geometric distribution.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the geometric distribution.

    .. seealso::
        :func:`numpy.random.geometric`
    """
    rs = _generator.get_random_state()
    return rs.geometric(p, size, dtype)