import warnings
from scipy import linalg, special
from scipy._lib._util import check_random_state
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, exp, pi,
import numpy as np
from . import _mvn
from ._stats import gaussian_kernel_estimate, gaussian_kernel_estimate_log
from scipy.special import logsumexp  # noqa: F401
@property
def inv_cov(self):
    self.factor = self.covariance_factor()
    self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
    return linalg.inv(self._data_covariance) / self.factor ** 2