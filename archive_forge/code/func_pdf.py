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
@inherit_docstring_from(rv_continuous)
def pdf(self, x, *args, **kwds):
    if self._parameterization() == 'S0':
        return super().pdf(x, *args, **kwds)
    elif self._parameterization() == 'S1':
        (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
        if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
            return super().pdf(x, *args, **kwds)
        else:
            x = np.reshape(x, (1, -1))[0, :]
            x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
            data_in = np.dstack((x, alpha, beta))[0]
            data_out = np.empty(shape=(len(data_in), 1))
            uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
            for pair in uniq_param_pairs:
                _alpha, _beta = pair
                _delta = delta + 2 * _beta * gamma * np.log(gamma) / np.pi if _alpha == 1.0 else delta
                data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                _x = data_in[data_mask, 0]
                data_out[data_mask] = super().pdf(_x, _alpha, _beta, loc=_delta, scale=gamma).reshape(len(_x), 1)
            output = data_out.T[0]
            if output.shape == (1,):
                return output[0]
            return output