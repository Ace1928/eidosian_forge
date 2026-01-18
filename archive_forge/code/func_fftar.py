import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def fftar(self, n=None):
    """Fourier transform of AR polynomial, zero-padded at end to n

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftar : ndarray
            fft of zero-padded ar polynomial
        """
    if n is None:
        n = len(self.ar)
    return fft.fft(self.padarr(self.ar, n))