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
def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
    val0 = beta * np.tan(np.pi * alpha / 2)
    th0 = np.arctan(val0) / alpha
    val3 = W / (cosTH / np.tan(alpha * (th0 + TH)) + np.sin(TH))
    res3 = val3 * ((np.cos(aTH) + np.sin(aTH) * tanTH - val0 * (np.sin(aTH) - np.cos(aTH) * tanTH)) / W) ** (1.0 / alpha)
    return res3