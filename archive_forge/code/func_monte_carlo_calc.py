from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def monte_carlo_calc(n_samples):
    """
        Calculate function(x, y) on n_samples samples and returns the
        median, and the covariances between (x, y, function(x, y)).
        """
    x_samples = numpy.random.normal(x.nominal_value, x.std_dev, n_samples)
    y_samples = numpy.random.normal(y.nominal_value, y.std_dev, n_samples)
    function_samples = function(x_samples, y_samples).astype(float)
    cov_mat = numpy.cov([x_samples, y_samples], function_samples)
    return (numpy.median(function_samples), cov_mat)