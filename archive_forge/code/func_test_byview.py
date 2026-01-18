import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_byview(self):
    base = self.base
    mbase = base.view(mrecarray)
    assert_equal(mbase.recordmask, base.recordmask)
    assert_equal_records(mbase._mask, base._mask)
    assert_(isinstance(mbase._data, recarray))
    assert_equal_records(mbase._data, base._data.view(recarray))
    for field in ('a', 'b', 'c'):
        assert_equal(base[field], mbase[field])
    assert_equal_records(mbase.view(mrecarray), mbase)