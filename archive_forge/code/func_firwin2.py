import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
def firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', nyq=None, antisymmetric=False, fs=2.0):
    """
    FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``. Default is 2.

    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.

    See Also
    --------
    scipy.signal.firwin2
    firls
    firwin
    minimum_phase
    remez

    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.
    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:

       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced

    Magnitude response of all but type I filters are subjects to following
    constraints:

       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency
    """
    nyq = 0.5 * fs
    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')
    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError('ntaps must be less than nfreqs, but firwin2 was called with ntaps=%d and nfreqs=%s' % (numtaps, nfreqs))
    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with fs/2.')
    d = cupy.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')
    if freq[1] == 0:
        raise ValueError('Value 0 must not be repeated in freq')
    if freq[-2] == nyq:
        raise ValueError('Value fs/2 must not be repeated in freq')
    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    elif numtaps % 2 == 0:
        ftype = 2
    else:
        ftype = 1
    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError('A Type II filter must have zero gain at the Nyquist frequency.')
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError('A Type III filter must have zero gain at zero and Nyquist frequencies.')
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError('A Type IV filter must have zero gain at zero frequency.')
    if nfreqs is None:
        nfreqs = 1 + 2 ** int(math.ceil(math.log(numtaps, 2)))
    if (d == 0).any():
        freq = cupy.array(freq, copy=True)
        eps = cupy.finfo(float).eps * nyq
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        d = cupy.diff(freq)
        if (d <= 0).any():
            raise ValueError('freq cannot contain numbers that are too close (within eps * (fs/2): {}) to a repeated value'.format(eps))
    x = cupy.linspace(0.0, nyq, nfreqs)
    fx = cupy.interp(x, freq, gain)
    shift = cupy.exp(-(numtaps - 1) / 2.0 * 1j * math.pi * x / nyq)
    if ftype > 2:
        shift *= 1j
    fx2 = fx * shift
    out_full = cupy.fft.irfft(fx2)
    if window is not None:
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1
    out = out_full[:numtaps] * wind
    if ftype == 3:
        out[out.size // 2] = 0.0
    return out