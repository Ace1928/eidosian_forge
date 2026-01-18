from statsmodels.compat.python import lzip, lmap
from numpy.testing import (
import numpy as np
import pytest
from statsmodels.stats.libqsturng import qsturng, psturng
@pytest.mark.slow
def test_10000_to_ch(self):
    import os
    curdir = os.path.dirname(os.path.abspath(__file__))
    ps, rs, vs, qs = read_ch(os.path.split(os.path.split(curdir)[0])[0] + '/tests/results/bootleg.csv')
    qs = np.array(qs)
    errors = np.abs(qs - qsturng(ps, rs, vs)) / qs
    assert_equal(np.array([]), np.where(errors > 0.03)[0])