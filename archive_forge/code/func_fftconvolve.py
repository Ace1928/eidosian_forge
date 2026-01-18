import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def fftconvolve(in1, in2, mode='full', axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve ``in1`` and ``in2`` using the fast Fourier transform method, with
    the output size determined by the ``mode`` argument.

    This is generally much faster than the ``'direct'`` method of ``convolve``
    for large arrays, but can be slower when only a few output values are
    needed, and can only output float arrays (int or object array inputs will
    be cast to float).

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear                           cross-correlation (default)
            - ``'valid'``: output consists only of those elements that do                            not rely on the zero-padding. Either ``in1`` or                            ``in2`` must be at least as large as the other in                            every dimension.
            - ``'same'``: output is the same size as ``in1``, centered                           with respect to the 'full' output

        axes (scalar or tuple of scalar or None): Axes over which to compute
            the convolution. The default is over all axes.

    Returns:
        cupy.ndarray: the result of convolution

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.correlation`
    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.correlation`
    """
    out = _st_core._check_conv_inputs(in1, in2, mode)
    if out is not None:
        return out
    in1, in2, axes = _st_core._init_freq_conv_axes(in1, in2, mode, axes, False)
    shape = [max(x1, x2) if a not in axes else x1 + x2 - 1 for a, (x1, x2) in enumerate(zip(in1.shape, in2.shape))]
    out = _st_core._freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)
    return _st_core._apply_conv_mode(out, in1.shape, in2.shape, mode, axes)