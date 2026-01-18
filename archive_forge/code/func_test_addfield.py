import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_addfield(self):
    mrec, nrec, ddtype = self.data
    d, m = ([100, 200, 300], [1, 0, 0])
    mrec = addfield(mrec, ma.array(d, mask=m))
    assert_equal(mrec.f3, d)
    assert_equal(mrec.f3._mask, m)