from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def fit_loc_scale(self, data, *args):
    """
        Estimate loc and scale parameters from data using 1st and 2nd moments.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
    mu, mu2 = self.stats(*args, **{'moments': 'mv'})
    tmp = asarray(data)
    muhat = tmp.mean()
    mu2hat = tmp.var()
    Shat = sqrt(mu2hat / mu2)
    with np.errstate(invalid='ignore'):
        Lhat = muhat - Shat * mu
    if not np.isfinite(Lhat):
        Lhat = 0
    if not (np.isfinite(Shat) and 0 < Shat):
        Shat = 1
    return (Lhat, Shat)