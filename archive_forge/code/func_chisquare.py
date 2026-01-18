from cupy.random import _generator
from cupy import _util
def chisquare(df, size=None, dtype=float):
    """Chi-square distribution.

    Returns an array of samples drawn from the chi-square distribution. Its
    probability density function is defined as

    .. math::
       f(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}x^{k/2-1}e^{-x/2}.

    Args:
        df (int or array_like of ints): Degree of freedom :math:`k`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the chi-square distribution.

    .. seealso::
        :func:`numpy.random.chisquare`
    """
    rs = _generator.get_random_state()
    return rs.chisquare(df, size, dtype)