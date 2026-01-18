from statsmodels.compat.numpy import NP_LT_2
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from scipy import linalg, optimize, signal
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.validation import array_like
@classmethod
def from_coeffs(cls, arcoefs=None, macoefs=None, nobs=100):
    """
        Create ArmaProcess from an ARMA representation.

        Parameters
        ----------
        arcoefs : array_like
            Coefficient for autoregressive lag polynomial, not including zero
            lag. The sign is inverted to conform to the usual time series
            representation of an ARMA process in statistics. See the class
            docstring for more information.
        macoefs : array_like
            Coefficient for moving-average lag polynomial, excluding zero lag.
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Returns
        -------
        ArmaProcess
            Class instance initialized with arcoefs and macoefs.

        Examples
        --------
        >>> arparams = [.75, -.25]
        >>> maparams = [.65, .35]
        >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(ar, ma)
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
    arcoefs = [] if arcoefs is None else arcoefs
    macoefs = [] if macoefs is None else macoefs
    return cls(np.r_[1, -np.asarray(arcoefs)], np.r_[1, np.asarray(macoefs)], nobs=nobs)