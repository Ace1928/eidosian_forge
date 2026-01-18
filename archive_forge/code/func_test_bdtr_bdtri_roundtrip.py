import numpy as np
import scipy.special as sc
import pytest
from numpy.testing import assert_allclose, assert_array_equal, suppress_warnings
def test_bdtr_bdtri_roundtrip(self):
    bdtr_vals = sc.bdtr([0, 1, 2], 2, 0.5)
    roundtrip_vals = sc.bdtri([0, 1, 2], 2, bdtr_vals)
    assert_allclose(roundtrip_vals, [0.5, 0.5, np.nan])