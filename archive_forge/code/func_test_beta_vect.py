import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import heart
from statsmodels.emplike.aft_el import emplikeAFT
from statsmodels.tools import add_constant
from .results.el_results import AFTRes
def test_beta_vect(self):
    assert_almost_equal(self.res1.test_beta([3.5, -0.035], [0, 1]), self.res2.test_joint, decimal=4)