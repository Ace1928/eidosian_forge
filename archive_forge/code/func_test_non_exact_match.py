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
def test_non_exact_match(self):
    arr = np.array([[3, 6, 6], [4, 5, 1]])
    self.assert_deprecated(lambda: np.ravel_multi_index(arr, (7, 6), mode='Cilp'))
    self.assert_deprecated(lambda: np.searchsorted(arr[0], 4, side='Random'))