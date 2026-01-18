import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def oaconvolve(in1, in2, mode='full', axes=None):
    """Convolve two N-dimensional arrays using the overlap-add method.

    Convolve ``in1`` and ``in2`` using the overlap-add method, with the output
    size determined by the ``mode`` argument. This is generally faster than
    ``convolve`` for large arrays, and generally faster than ``fftconvolve``
    when one array is much larger than the other, but can be slower when only a
    few output values are needed or when the arrays are very similar in shape,
    and can only output float arrays (int or object array inputs will be cast
    to float).

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear                           cross-correlation (default)
            - ``'valid'``: output consists only of those elements that do                            not rely on the zero-padding. Either ``in1`` or                            ``in2`` must be at least as large as the other in                            every dimension.
            - ``'same'``: output is the same size as ``in1``, centered                           with respect to the ``'full'`` output

        axes (scalar or tuple of scalar or None): Axes over which to compute
            the convolution. The default is over all axes.

    Returns:
        cupy.ndarray: the result of convolution

    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.oaconvolve`
    """
    out = _st_core._check_conv_inputs(in1, in2, mode)
    if out is not None:
        return out
    if in1.shape == in2.shape:
        return fftconvolve(in1, in2, mode=mode, axes=axes)
    in1, in2, axes = _st_core._init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=True)
    s1, s2 = (in1.shape, in2.shape)
    if not axes:
        return _st_core._apply_conv_mode(in1 * in2, s1, s2, mode, axes)
    optimal_sizes = (_st_core._calc_oa_lens(s1[i], s2[i]) if i in axes else (-1, -1, s1[i], s2[i]) for i in range(in1.ndim))
    block_size, overlaps, in1_step, in2_step = zip(*optimal_sizes)
    if in1_step == s1 and in2_step == s2:
        return fftconvolve(in1, in2, mode=mode, axes=axes)
    shape_final = [s1[i] + s2[i] - 1 if i in axes else None for i in range(in1.ndim)]
    in1, in2 = _st_core._oa_reshape_inputs(in1, in2, axes, shape_final, block_size, overlaps, in1_step, in2_step)
    split_axes = [iax + i for i, iax in enumerate(axes)]
    fft_axes = [iax + 1 for iax in split_axes]
    fft_shape = [block_size[i] for i in axes]
    ret = _st_core._freq_domain_conv(in1, in2, fft_axes, fft_shape, calc_fast_len=False)
    for ax, ax_fft, ax_split in zip(axes, fft_axes, split_axes):
        overlap = overlaps[ax]
        if overlap is None:
            continue
        ret, overpart = cupy.split(ret, [-overlap], ax_fft)
        overpart = cupy.split(overpart, [-1], ax_split)[0]
        ret_overpart = cupy.split(ret, [overlap], ax_fft)[0]
        ret_overpart = cupy.split(ret_overpart, [1], ax_split)[1]
        ret_overpart += overpart
    shape_ret = [ret.shape[i] if i not in fft_axes else ret.shape[i] * ret.shape[i - 1] for i in range(ret.ndim) if i not in split_axes]
    ret = ret.reshape(*shape_ret)
    ret = ret[tuple([slice(islice) for islice in shape_final])]
    return _st_core._apply_conv_mode(ret, s1, s2, mode, axes)