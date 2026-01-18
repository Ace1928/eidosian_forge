import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
@_fft._implements(_fft._scipy_fft.dctn)
def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
    """Compute a multidimensional Discrete Cosine Transform.

    Parameters
    ----------
    x : cupy.ndarray
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : cupy.ndarray of real
        The transformed input array.

    See Also
    --------
    :func:`scipy.fft.dctn`

    Notes
    -----
    For full details of the DCT types and normalization modes, as well as
    references, see `dct`.
    """
    if x.dtype.kind == 'c':
        out = dctn(x.real, type, s, axes, norm, overwrite_x)
        out = out + 1j * dctn(x.imag, type, s, axes, norm, overwrite_x)
        return out
    shape, axes = _init_nd_shape_and_axes(x, s, axes)
    x = _promote_dtype(x)
    if len(axes) == 0:
        return x
    for n, axis in zip(shape, axes):
        x = dct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)
    return x