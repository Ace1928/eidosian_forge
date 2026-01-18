import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def test_NAAction_raise():
    action = NAAction(on_NA='raise')
    in_arrs = [np.asarray([1.1, 1.2]), np.asarray([1, 2])]
    is_NAs = [np.asarray([False, False])] * 2
    got_arrs = action.handle_NA(in_arrs, is_NAs, [None, None])
    assert np.array_equal(got_arrs[0], in_arrs[0])
    assert np.array_equal(got_arrs[1], in_arrs[1])
    from patsy.origin import Origin
    o1 = Origin('asdf', 0, 1)
    o2 = Origin('asdf', 2, 3)
    in_idx = np.arange(2)
    in_arrs = [np.asarray([1.1, 1.2]), np.asarray([1.0, np.nan])]
    is_NAs = [np.asarray([False, False]), np.asarray([False, True])]
    try:
        action.handle_NA(in_arrs, is_NAs, [o1, o2])
        assert False
    except PatsyError as e:
        assert e.origin is o2