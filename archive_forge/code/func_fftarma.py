import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def fftarma(self, n=None):
    """Fourier transform of ARMA polynomial, zero-padded at end to n

        The Fourier transform of the ARMA process is calculated as the ratio
        of the fft of the MA polynomial divided by the fft of the AR polynomial.

        Parameters
        ----------
        n : int
            length of array after zero-padding

        Returns
        -------
        fftarma : ndarray
            fft of zero-padded arma polynomial
        """
    if n is None:
        n = self.nobs
    return self.fftma(n) / self.fftar(n)