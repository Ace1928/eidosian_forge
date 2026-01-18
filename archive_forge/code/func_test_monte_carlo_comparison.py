from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def test_monte_carlo_comparison():
    """
    Full comparison to a Monte-Carlo calculation.

    Both the nominal values and the covariances are compared between
    the direct calculation performed in this module and a Monte-Carlo
    simulation.
    """
    try:
        import numpy
        import numpy.random
    except ImportError:
        import warnings
        warnings.warn('Test not performed because NumPy is not available')
        return
    sin_uarray_uncert = numpy.vectorize(umath_core.sin, otypes=[object])

    def function(x, y):
        """
        Function that takes two NumPy arrays of the same size.
        """
        return 10 * x ** 2 - x * sin_uarray_uncert(y ** 3)
    x = ufloat(0.2, 0.01)
    y = ufloat(10, 0.001)
    function_result_this_module = function(x, y)
    nominal_value_this_module = function_result_this_module.nominal_value
    covariances_this_module = numpy.array(uncert_core.covariance_matrix((x, y, function_result_this_module)))

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
    nominal_value_samples, covariances_samples = monte_carlo_calc(1000000)
    assert numpy.vectorize(test_uncertainties.numbers_close)(covariances_this_module, covariances_samples, 0.06).all(), 'The covariance matrices do not coincide between the Monte-Carlo simulation and the direct calculation:\n* Monte-Carlo:\n%s\n* Direct calculation:\n%s' % (covariances_samples, covariances_this_module)
    assert test_uncertainties.numbers_close(nominal_value_this_module, nominal_value_samples, math.sqrt(covariances_samples[2, 2]) / abs(nominal_value_samples) * 0.5), 'The nominal value (%f) does not coincide with that of the Monte-Carlo simulation (%f), for a standard deviation of %f.' % (nominal_value_this_module, nominal_value_samples, math.sqrt(covariances_samples[2, 2]))