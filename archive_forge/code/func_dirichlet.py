from cupy.random import _generator
from cupy import _util
def dirichlet(alpha, size=None, dtype=float):
    """Dirichlet distribution.

    Returns an array of samples drawn from the dirichlet distribution. Its
    probability density function is defined as

    .. math::
        f(x) = \\frac{\\Gamma(\\sum_{i=1}^K\\alpha_i)}             {\\prod_{i=1}^{K}\\Gamma(\\alpha_i)}             \\prod_{i=1}^Kx_i^{\\alpha_i-1}.

    Args:
        alpha (array): Parameters of the dirichlet distribution
            :math:`\\alpha`.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the dirichlet distribution.

    .. seealso::
        :func:`numpy.random.dirichlet`
    """
    rs = _generator.get_random_state()
    return rs.dirichlet(alpha, size, dtype)