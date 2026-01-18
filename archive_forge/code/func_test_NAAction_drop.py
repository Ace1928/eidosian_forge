import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def test_NAAction_drop():
    action = NAAction('drop')
    in_values = [np.asarray([-1, 2, -1, 4, 5]), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), np.asarray([[1.0, np.nan], [3.0, 4.0], [10.0, 5.0], [6.0, 7.0], [8.0, np.nan]])]
    is_NAs = [np.asarray([True, False, True, False, False]), np.zeros(5, dtype=bool), np.asarray([True, False, False, False, True])]
    out_values = action.handle_NA(in_values, is_NAs, [None] * 3)
    assert len(out_values) == 3
    assert np.array_equal(out_values[0], [2, 4])
    assert np.array_equal(out_values[1], [20.0, 40.0])
    assert np.array_equal(out_values[2], [[3.0, 4.0], [6.0, 7.0]])