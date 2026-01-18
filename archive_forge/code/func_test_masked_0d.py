import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_0d(self):
    x = array(1, mask=False)
    assert_equal(np.ma.median(x), 1)
    x = array(1, mask=True)
    assert_equal(np.ma.median(x), np.ma.masked)