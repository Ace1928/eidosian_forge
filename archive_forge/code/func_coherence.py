import warnings
import cupy
from cupyx.scipy.signal.windows._windows import get_window
from cupyx.scipy.signal._spectral_impl import (
def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', axis=-1):
    """
    Estimate the magnitude squared coherence estimate, Cxy, of
    discrete-time signals X and Y using Welch's method.

    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power
    spectral density estimates of X and Y, and `Pxy` is the cross
    spectral density estimate of X and Y.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    axis : int, optional
        Axis along which the coherence is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Cxy : ndarray
        Magnitude squared coherence of x and y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap. See [1]_ and [2]_ for more
    information.

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
           Signals" Prentice Hall, 2005

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy.signal import butter, lfilter, coherence
    >>> import matplotlib.pyplot as plt

    Generate two test signals with some common features.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = cupy.arange(N) / fs
    >>> b, a = butter(2, 0.25, 'low')
    >>> x = cupy.random.normal(
    ...         scale=cupy.sqrt(noise_power), size=time.shape)
    >>> y = lfilter(b, a, x)
    >>> x += amp * cupy.sin(2*cupy.pi*freq*time)
    >>> y += cupy.random.normal(
    ...         scale=0.1*cupy.sqrt(noise_power), size=time.shape)

    Compute and plot the coherence.

    >>> f, Cxy = coherence(x, y, fs, nperseg=1024)
    >>> plt.semilogy(cupy.asnumpy(f), cupy.asnumpy(Cxy))
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Coherence')
    >>> plt.show()
    """
    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    _, Pyy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    Cxy = cupy.abs(Pxy) ** 2 / Pxx / Pyy
    return (freqs, Cxy)