from cupy.random import _generator
from cupy import _util
def pareto(a, size=None, dtype=float):
    """Pareto II or Lomax distribution.

    Returns an array of samples drawn from the Pareto II distribution. Its
    probability density function is defined as

    .. math::
        f(x) = \\alpha(1+x)^{-(\\alpha+1)}.

    Args:
        a (float or array_like of floats): Parameter of the Pareto II
            distribution :math:`\\alpha`.
        size (int or tuple of ints): The shape of the array. If ``None``, this
            function generate an array whose shape is `a.shape`.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the Pareto II distribution.

    .. seealso:: :func:`numpy.random.pareto`
    """
    rs = _generator.get_random_state()
    return rs.pareto(a, size, dtype)