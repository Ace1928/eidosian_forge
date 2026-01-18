from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
@pytest.fixture(scope='class', autouse=True)
def stim(self, request, fstims, iscomplex, sides, len_x, NFFT_density, nover_density, pad_to_density, pad_to_spectrum):
    Fs = 100.0
    x = np.arange(0, 10, 1 / Fs)
    if len_x is not None:
        x = x[:len_x]
    fstims = [Fs / fstim for fstim in fstims]
    if NFFT_density is None:
        NFFT_density_real = 256
    elif NFFT_density < 0:
        NFFT_density_real = NFFT_density = 100
    else:
        NFFT_density_real = NFFT_density
    if nover_density is None:
        nover_density_real = 0
    elif nover_density < 0:
        nover_density_real = nover_density = NFFT_density_real // 2
    else:
        nover_density_real = nover_density
    if pad_to_density is None:
        pad_to_density_real = NFFT_density_real
    elif pad_to_density < 0:
        pad_to_density = int(2 ** np.ceil(np.log2(NFFT_density_real)))
        pad_to_density_real = pad_to_density
    else:
        pad_to_density_real = pad_to_density
    if pad_to_spectrum is None:
        pad_to_spectrum_real = len(x)
    elif pad_to_spectrum < 0:
        pad_to_spectrum_real = pad_to_spectrum = len(x)
    else:
        pad_to_spectrum_real = pad_to_spectrum
    if pad_to_spectrum is None:
        NFFT_spectrum_real = NFFT_spectrum = pad_to_spectrum_real
    else:
        NFFT_spectrum_real = NFFT_spectrum = len(x)
    nover_spectrum = 0
    NFFT_specgram = NFFT_density
    nover_specgram = nover_density
    pad_to_specgram = pad_to_density
    NFFT_specgram_real = NFFT_density_real
    nover_specgram_real = nover_density_real
    if sides == 'onesided' or (sides == 'default' and (not iscomplex)):
        if pad_to_density_real % 2:
            freqs_density = np.linspace(0, Fs / 2, num=pad_to_density_real, endpoint=False)[::2]
        else:
            freqs_density = np.linspace(0, Fs / 2, num=pad_to_density_real // 2 + 1)
        if pad_to_spectrum_real % 2:
            freqs_spectrum = np.linspace(0, Fs / 2, num=pad_to_spectrum_real, endpoint=False)[::2]
        else:
            freqs_spectrum = np.linspace(0, Fs / 2, num=pad_to_spectrum_real // 2 + 1)
    else:
        if pad_to_density_real % 2:
            freqs_density = np.linspace(-Fs / 2, Fs / 2, num=2 * pad_to_density_real, endpoint=False)[1::2]
        else:
            freqs_density = np.linspace(-Fs / 2, Fs / 2, num=pad_to_density_real, endpoint=False)
        if pad_to_spectrum_real % 2:
            freqs_spectrum = np.linspace(-Fs / 2, Fs / 2, num=2 * pad_to_spectrum_real, endpoint=False)[1::2]
        else:
            freqs_spectrum = np.linspace(-Fs / 2, Fs / 2, num=pad_to_spectrum_real, endpoint=False)
    freqs_specgram = freqs_density
    t_start = NFFT_specgram_real // 2
    t_stop = len(x) - NFFT_specgram_real // 2 + 1
    t_step = NFFT_specgram_real - nover_specgram_real
    t_specgram = x[t_start:t_stop:t_step]
    if NFFT_specgram_real % 2:
        t_specgram += 1 / Fs / 2
    if len(t_specgram) == 0:
        t_specgram = np.array([NFFT_specgram_real / (2 * Fs)])
    t_spectrum = np.array([NFFT_spectrum_real / (2 * Fs)])
    t_density = t_specgram
    y = np.zeros_like(x)
    for i, fstim in enumerate(fstims):
        y += np.sin(fstim * x * np.pi * 2) * 10 ** i
    if iscomplex:
        y = y.astype('complex')
    cls = request.cls
    cls.Fs = Fs
    cls.sides = sides
    cls.fstims = fstims
    cls.NFFT_density = NFFT_density
    cls.nover_density = nover_density
    cls.pad_to_density = pad_to_density
    cls.NFFT_spectrum = NFFT_spectrum
    cls.nover_spectrum = nover_spectrum
    cls.pad_to_spectrum = pad_to_spectrum
    cls.NFFT_specgram = NFFT_specgram
    cls.nover_specgram = nover_specgram
    cls.pad_to_specgram = pad_to_specgram
    cls.t_specgram = t_specgram
    cls.t_density = t_density
    cls.t_spectrum = t_spectrum
    cls.y = y
    cls.freqs_density = freqs_density
    cls.freqs_spectrum = freqs_spectrum
    cls.freqs_specgram = freqs_specgram
    cls.NFFT_density_real = NFFT_density_real