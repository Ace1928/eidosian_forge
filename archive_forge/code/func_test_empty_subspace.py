import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
@pytest.mark.parametrize('index', [([3, 0],), ([0, 0], [3, 0])])
def test_empty_subspace(self, index):
    arr = np.ones((2, 2, 0))
    self.assert_deprecated(arr.__getitem__, args=(index,))
    self.assert_deprecated(arr.__setitem__, args=(index, 0.0))
    arr2 = np.ones((2, 2, 1))
    index2 = (slice(0, 0),) + index
    self.assert_deprecated(arr2.__getitem__, args=(index2,))
    self.assert_deprecated(arr2.__setitem__, args=(index2, 0.0))