import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_fromrecords_wmask(self):
    mrec, nrec, ddtype = self.data
    _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=[0, 1, 0])
    assert_equal_records(_mrec._data, mrec._data)
    assert_equal(_mrec._mask.tolist(), [(0, 0, 0), (1, 1, 1), (0, 0, 0)])
    _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=True)
    assert_equal_records(_mrec._data, mrec._data)
    assert_equal(_mrec._mask.tolist(), [(1, 1, 1), (1, 1, 1), (1, 1, 1)])
    _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask)
    assert_equal_records(_mrec._data, mrec._data)
    assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())
    _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask.tolist())
    assert_equal_records(_mrec._data, mrec._data)
    assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())