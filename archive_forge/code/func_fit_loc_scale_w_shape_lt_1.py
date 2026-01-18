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
def fit_loc_scale_w_shape_lt_1():
    loc = np.nextafter(data.min(), -np.inf)
    if np.abs(loc) < np.finfo(loc.dtype).tiny:
        loc = np.sign(loc) * np.finfo(loc.dtype).tiny
    scale = np.nextafter(get_scale(data, loc), np.inf)
    shape = fshape or get_shape(data, loc, scale)
    return (shape, loc, scale)