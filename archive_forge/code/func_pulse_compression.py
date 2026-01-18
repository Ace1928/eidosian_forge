import cupy
from cupyx.scipy.signal import windows
def pulse_compression(x, template, normalize=False, window=None, nfft=None):
    """
    Pulse Compression is used to increase the range resolution and SNR
    by performing matched filtering of the transmitted pulse (template)
    with the received signal (x)

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    template : ndarray
        Transmitted signal, assume 1D array

    normalize : bool
        Normalize transmitted signal

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    compressedIQ : ndarray
        Pulse compressed output
    """
    num_pulses, samples_per_pulse = x.shape
    dtype = cupy.result_type(x, template)
    if nfft is None:
        nfft = samples_per_pulse
    t = _pulse_preprocess(template, normalize, window)
    fft_x = cupy.fft.fft(x, nfft)
    fft_t = cupy.fft.fft(t, nfft)
    out = cupy.fft.ifft(fft_x * fft_t.conj(), nfft)
    if dtype.kind != 'c':
        out = out.real
    return out