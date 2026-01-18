from cupy.random import _generator
from cupy import _util
def negative_binomial(n, p, size=None, dtype=int):
    """Negative binomial distribution.

    Returns an array of samples drawn from the negative binomial distribution.
    Its probability mass function is defined as

    .. math::
        f(x) = \\binom{x + n - 1}{n - 1}p^n(1-p)^{x}.

    Args:
        n (int): Parameter of the negative binomial distribution :math:`n`.
        p (float): Parameter of the negative binomial distribution :math:`p`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.int32` and
            :class:`numpy.int64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the negative binomial distribution.

    .. seealso::
        :func:`numpy.random.negative_binomial`
    """
    rs = _generator.get_random_state()
    return rs.negative_binomial(n, p, size=size, dtype=dtype)