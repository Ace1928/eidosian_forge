import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spdshift(self, n):
    """power spectral density using fftshift

        currently returns two-sided according to fft frequencies, use first half
        """
    mapadded = self.padarr(self.ma, n)
    arpadded = self.padarr(self.ar, n)
    hw = fft.fft(fft.fftshift(mapadded)) / fft.fft(fft.fftshift(arpadded))
    w = fft.fftfreq(n) * 2 * np.pi
    wslice = slice(n // 2 - 1, None, None)
    return ((hw * hw.conj()).real, w)