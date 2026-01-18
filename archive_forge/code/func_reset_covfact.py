import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt
def reset_covfact(self, covfact):
    self.covfact = covfact
    self.covariance_factor()
    self._compute_covariance()