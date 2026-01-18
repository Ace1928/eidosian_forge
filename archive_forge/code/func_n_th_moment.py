import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
def n_th_moment(n, beta, m):
    """
            Returns n-th moment. Defined only if n+1 < m
            Function cannot broadcast due to the loop over n
            """
    A = (m / beta) ** m * np.exp(-beta ** 2 / 2.0)
    B = m / beta - beta
    rhs = 2 ** ((n - 1) / 2.0) * sc.gamma((n + 1) / 2) * (1.0 + (-1) ** n * sc.gammainc((n + 1) / 2, beta ** 2 / 2))
    lhs = np.zeros(rhs.shape)
    for k in range(n + 1):
        lhs += sc.binom(n, k) * B ** (n - k) * (-1) ** k / (m - k - 1) * (m / beta) ** (-m + k + 1)
    return A * lhs + rhs