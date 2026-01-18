from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import stackloss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from .results.el_results import RegressionResults
@pytest.mark.slow
def test_hypothesis_beta3(self):
    beta3res = self.res1.el_test([0], [3], return_weights=1, method='nm')
    assert_almost_equal(beta3res[:2], self.res2.test_beta3[:2], 4)
    assert_almost_equal(beta3res[2], self.res2.test_beta3[2], 4)