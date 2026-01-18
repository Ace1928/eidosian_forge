import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.tools.eval_measures import (
@pytest.mark.parametrize('ic,ic_sig', zip(ics, ics_sig))
def test_ic_equivalence(ic, ic_sig):
    assert ic(np.array(2), 10, 2).dtype == float
    assert ic_sig(np.array(2), 10, 2).dtype == float
    assert_almost_equal(ic(-10.0 / 2.0 * np.log(2.0), 10, 2) / 10, ic_sig(2, 10, 2), decimal=14)
    assert_almost_equal(ic_sig(np.log(2.0), 10, 2, islog=True), ic_sig(2, 10, 2), decimal=14)