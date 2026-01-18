import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
def test_coefficients(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ne3 = ExpandedNormal([0.0, 1.0, 1.0])
        assert_allclose(ne3._coef, [1.0, 0.0, 0.0, 1.0 / 6])
        ne4 = ExpandedNormal([0.0, 1.0, 1.0, 1.0])
        assert_allclose(ne4._coef, [1.0, 0.0, 0.0, 1.0 / 6, 1.0 / 24, 0.0, 1.0 / 72])
        ne5 = ExpandedNormal([0.0, 1.0, 1.0, 1.0, 1.0])
        assert_allclose(ne5._coef, [1.0, 0.0, 0.0, 1.0 / 6, 1.0 / 24, 1.0 / 120, 1.0 / 72, 1.0 / 144, 0.0, 1.0 / 1296])
        ne33 = ExpandedNormal([0.0, 1.0, 1.0, 0.0])
        assert_allclose(ne33._coef, [1.0, 0.0, 0.0, 1.0 / 6, 0.0, 0.0, 1.0 / 72])