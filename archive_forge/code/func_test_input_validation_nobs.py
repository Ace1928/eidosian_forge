import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_input_validation_nobs(self):
    with pytest.raises(ValueError, match='same number of observations'):
        ddm.distance_covariance_test(self.x[:2, :], self.y)