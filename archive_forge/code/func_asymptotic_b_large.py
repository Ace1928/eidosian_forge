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
def asymptotic_b_large(a, b):
    sum_ab = a + b
    t1 = sc.gammaln(a) - (a - 1) * sc.psi(a)
    t2 = -1 / (2 * b) + 1 / (12 * b) - b ** (-2.0) / 12 - b ** (-3.0) / 120 + b ** (-4.0) / 120 + b ** (-5.0) / 252 - b ** (-6.0) / 252 + 1 / sum_ab - 1 / (12 * sum_ab) + sum_ab ** (-2.0) / 6 + sum_ab ** (-3.0) / 120 - sum_ab ** (-4.0) / 60 - sum_ab ** (-5.0) / 252 + sum_ab ** (-6.0) / 126
    log_term = sum_ab * np.log1p(a / b) + np.log(b) - 2 * np.log(sum_ab)
    return t1 + t2 + log_term