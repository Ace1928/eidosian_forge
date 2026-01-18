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
def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the sample positions
        associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:

        ``time``
           Consider the input `x` as time-domain. (Default)
        ``freq``
           Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `cusignal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it represents the old sample positions,
    and the new sample positions will be returned as well as the new
    samples.

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fftpack.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> import cupy as cp
    >>> import cupyx.scipy.signal import resample

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f = resample(y, 100)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cupy.asnumpy(x), cupy.asnumpy(y), 'go-', cupy.asnumpy(xnew),                 cupy.asnumpy(f), '.-', 10, cupy.asnumpy(y[0]), 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()
    """
    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or 'freq', not domain={}".format(domain))
    x = cupy.asarray(x)
    Nx = x.shape[axis]
    real_input = cupy.isrealobj(x)
    if domain == 'time':
        if real_input:
            X = rfft(x, axis=axis)
        else:
            X = fft(x, axis=axis)
    else:
        X = x
    if window is not None:
        if callable(window):
            W = window(fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            W = ifftshift(get_window(window, Nx))
        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = cupy.zeros(newshape, X.dtype)
    N = min(num, Nx)
    nyq = N // 2 + 1
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        if N > 2:
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]
    if N % 2 == 0:
        if num < Nx:
            if real_input:
                sl[axis] = slice(N // 2, N // 2 + 1)
                Y[tuple(sl)] *= 2.0
            else:
                sl[axis] = slice(-N // 2, -N // 2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:
            sl[axis] = slice(N // 2, N // 2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                sl[axis] = slice(num - N // 2, num - N // 2 + 1)
                Y[tuple(sl)] = temp
    if real_input:
        y = irfft(Y, num, axis=axis)
    else:
        y = ifft(Y, axis=axis, overwrite_x=True)
    y *= float(num) / float(Nx)
    if t is None:
        return y
    else:
        new_t = cupy.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return (y, new_t)