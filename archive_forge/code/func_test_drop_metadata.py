import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
def test_drop_metadata():

    def _compare_dtypes(dt1, dt2):
        return np.can_cast(dt1, dt2, casting='no')
    dt = np.dtype([('l1', [('l2', np.dtype('S8', metadata={'msg': 'toto'}))])], metadata={'msg': 'titi'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None
    assert dt_m['l1'].metadata is None
    assert dt_m['l1']['l2'].metadata is None
    dt = np.dtype([('x', '<f8'), ('y', '<i4')], align=True, metadata={'msg': 'toto'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None
    dt = np.dtype('8f', metadata={'msg': 'toto'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None
    dt = np.dtype('uint32', metadata={'msg': 'toto'})
    dt_m = utils.drop_metadata(dt)
    assert _compare_dtypes(dt, dt_m) is True
    assert dt_m.metadata is None