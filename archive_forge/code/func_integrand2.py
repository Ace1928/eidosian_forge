import warnings
from functools import partial
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.integrate._quadrature import _builtincoeffs
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import scipy.special as sc
from scipy._lib._util import _lazywhere
from .._distn_infrastructure import rv_continuous, _ShapeInfo
from .._continuous_distns import uniform, expon, _norm_pdf, _norm_cdf
from .levyst import Nolan
from scipy._lib.doccer import inherit_docstring_from
def integrand2(t):
    if t == 0:
        return 0
    return np.exp(-t ** alpha) * np.sin(beta * t ** alpha * Phi(alpha, t))