import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_dcov(self):
    assert_almost_equal(ddm.distance_covariance(self.x, self.y), self.dcov_exp, 4)