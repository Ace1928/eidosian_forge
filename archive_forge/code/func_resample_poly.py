import operator
from math import gcd
import cupy
from cupyx.scipy.fft import fft, rfft, fftfreq, ifft, irfft, ifftshift
from cupyx.scipy.signal._iir_filter_design import cheby1
from cupyx.scipy.signal._fir_filter_design import firwin
from cupyx.scipy.signal._iir_filter_conversions import zpk2sos
from cupyx.scipy.signal._ltisys import dlti
from cupyx.scipy.signal._upfirdn import upfirdn, _output_len
from cupyx.scipy.signal._signaltools import (
from cupyx.scipy.signal.windows._windows import get_window
def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0), padtype='constant', cval=None):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. Values beyond the boundary of the signal are assumed to be zero
    during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by
        `cupyx.scipy.signal.upfirdn`. Changes assumptions on values beyond
        the boundary. If `constant`, assumed to be `cval` (default zero).
        If `line` assumed to continue a linear trend defined by the first and
        last points. `mean`, `median`, `maximum` and `minimum` work as in
        `cupy.pad` and assume that the values beyond the boundary are the mean,
        median, maximum or minimum respectively of the array along the axis.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `cusignal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `cusignal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `cusignal.get_window`
    and `cusignal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> import cupy
    >>> import cupyx.scipy.signal import resample, resample_poly

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f_fft = resample(y, 100)
    >>> f_poly = resample_poly(y, 100, 20)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cupy.asnumpy(xnew), cupy.asnumpy(f_fft), 'b.-',                  cupy.asnumpy(xnew), cupy.asnumpy(f_poly), 'r.-')
    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'ko-')
    >>> plt.plot(10, cupy.asnumpy(y[0]), 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()
    """
    if padtype != 'constant' or cval is not None:
        raise ValueError('padtype and cval arguments are not supported by upfirdn')
    x = cupy.asarray(x)
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_out = x.shape[axis] * up
    n_out = n_out // down + bool(n_out % down)
    if isinstance(window, (list, cupy.ndarray)):
        window = cupy.asarray(window)
        if window.ndim > 1:
            raise ValueError('window must be 1-D')
        half_len = (window.size - 1) // 2
        h = up * window
    else:
        half_len = 10 * max(up, down)
        h = up * _design_resample_poly(up, down, window)
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis], up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = cupy.concatenate((cupy.zeros(n_pre_pad, h.dtype), h, cupy.zeros(n_post_pad, h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out
    y = upfirdn(h, x, up, down, axis)
    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    return y[tuple(keep)]