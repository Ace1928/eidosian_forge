import numpy as np
import numpy.fft as fft
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess
def padarr(self, arr, maxlag, atend=True):
    """pad 1d array with zeros at end to have length maxlag
        function that is a method, no self used

        Parameters
        ----------
        arr : array_like, 1d
            array that will be padded with zeros
        maxlag : int
            length of array after padding
        atend : bool
            If True (default), then the zeros are added to the end, otherwise
            to the front of the array

        Returns
        -------
        arrp : ndarray
            zero-padded array

        Notes
        -----
        This is mainly written to extend coefficient arrays for the lag-polynomials.
        It returns a copy.

        """
    if atend:
        return np.r_[arr, np.zeros(maxlag - len(arr))]
    else:
        return np.r_[np.zeros(maxlag - len(arr)), arr]