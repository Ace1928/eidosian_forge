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
def isf(self, q, *args, **kwds):
    """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
    args, loc, _ = self._parse_args(*args, **kwds)
    q, loc = map(asarray, (q, loc))
    args = tuple(map(asarray, args))
    _a, _b = self._get_support(*args)
    cond0 = self._argcheck(*args) & (loc == loc)
    cond1 = (q > 0) & (q < 1)
    cond2 = (q == 1) & cond0
    cond3 = (q == 0) & cond0
    cond = cond0 & cond1
    output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
    lower_bound = _a - 1 + loc
    upper_bound = _b + loc
    place(output, cond2 * (cond == cond), lower_bound)
    place(output, cond3 * (cond == cond), upper_bound)
    if np.any(cond):
        goodargs = argsreduce(cond, *(q,) + args + (loc,))
        loc, goodargs = (goodargs[-1], goodargs[:-1])
        place(output, cond, self._isf(*goodargs) + loc)
    if output.ndim == 0:
        return output[()]
    return output