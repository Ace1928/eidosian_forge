import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import cancer
from statsmodels.emplike.originregress import ELOriginRegress
from .results.el_results import OriginResults
def test_hypothesis_beta1(self):
    assert_almost_equal(self.res1.el_test([0.0034], [1])[0], self.res2.test_llf_hypoth, 4)