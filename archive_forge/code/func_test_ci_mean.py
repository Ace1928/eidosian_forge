import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_ci_mean(self):
    assert_almost_equal(self.res1.ci_mean(), self.res2.ci_mean, 4)