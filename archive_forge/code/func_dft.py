import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('dft')
def dft(n, scale=None):
    """Discrete Fourier transform matrix.

    Create the matrix that computes the discrete Fourier transform of a
    sequence. The nth primitive root of unity used to generate the matrix is
    exp(-2*pi*i/n), where i = sqrt(-1).

    Args:
        n (int): Size the matrix to create.
        scale (str, optional): Must be None, 'sqrtn', or 'n'.
            If ``scale`` is 'sqrtn', the matrix is divided by ``sqrt(n)``.
            If ``scale`` is 'n', the matrix is divided by ``n``.
            If ``scale`` is None (default), the matrix is not normalized, and
            the return value is simply the Vandermonde matrix of the roots of
            unity.

    Returns:
        (cupy.ndarray): The DFT matrix.

    Notes:
        When ``scale`` is None, multiplying a vector by the matrix returned by
        ``dft`` is mathematically equivalent to (but much less efficient than)
        the calculation performed by ``scipy.fft.fft``.

    .. seealso:: :func:`scipy.linalg.dft`
    """
    if scale not in (None, 'sqrtn', 'n'):
        raise ValueError("scale must be None, 'sqrtn', or 'n'; %r is not valid." % (scale,))
    r = cupy.arange(n, dtype='complex128')
    r *= -2j * cupy.pi / n
    omegas = cupy.exp(r, out=r)[:, None]
    m = omegas ** cupy.arange(n)
    if scale is not None:
        m *= 1 / math.sqrt(n) if scale == 'sqrtn' else 1 / n
    return m