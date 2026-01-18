from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import stackloss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from .results.el_results import RegressionResults
def test_hypothesis_beta0(self):
    beta0res = self.res1.el_test([-30], [0], return_weights=1, method='nm')
    assert_almost_equal(beta0res[:2], self.res2.test_beta0[:2], 4)
    assert_almost_equal(beta0res[2], self.res2.test_beta0[2], 4)