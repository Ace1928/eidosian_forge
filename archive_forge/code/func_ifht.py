import math
from warnings import warn
import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch
@_fft._implements(_scipy_fft.ifht)
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    """Compute the inverse fast Hankel transform.

    Computes the discrete inverse Hankel transform of a logarithmically spaced
    periodic sequence. This is the inverse operation to `fht`.

    Parameters
    ----------
    A : cupy.ndarray (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    a : cupy.ndarray (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.

    See Also
    --------
    :func:`scipy.special.ifht`
    :func:`scipy.special.fhtoffset` : Return an optimal offset for `fht`.

    """
    n = A.shape[-1]
    if bias != 0:
        j_c = (n - 1) / 2
        j = cupy.arange(n)
        A = A * cupy.exp(bias * ((j - j_c) * dln + offset))
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    a = _fhtq(A, u, inverse=True)
    if bias != 0:
        a /= cupy.exp(-bias * (j - j_c) * dln)
    return a