import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def spdroots(self, w):
    """spectral density for frequency using polynomial roots

        builds two arrays (number of roots, number of frequencies)
        """
    return self._spdroots(self.arroots, self.maroots, w)