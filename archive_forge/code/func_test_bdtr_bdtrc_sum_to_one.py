import numpy as np
import scipy.special as sc
import pytest
from numpy.testing import assert_allclose, assert_array_equal, suppress_warnings
def test_bdtr_bdtrc_sum_to_one(self):
    bdtr_vals = sc.bdtr([0, 1, 2], 2, 0.5)
    bdtrc_vals = sc.bdtrc([0, 1, 2], 2, 0.5)
    vals = bdtr_vals + bdtrc_vals
    assert_allclose(vals, [1.0, 1.0, 1.0])