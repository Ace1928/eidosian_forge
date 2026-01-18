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
@pytest.mark.parametrize('invalid_str', [',invalid_data', 'invalid_sep'])
def test_deprecate_unparsable_string(self, invalid_str):
    x = np.array([1.51, 2, 3.51, 4], dtype=float)
    x_str = '1.51,2,3.51,4{}'.format(invalid_str)
    self.assert_deprecated(lambda: np.fromstring(x_str, sep=','))
    self.assert_deprecated(lambda: np.fromstring(x_str, sep=',', count=5))
    bytestr = x_str.encode('ascii')
    self.assert_deprecated(lambda: fromstring_null_term_c_api(bytestr))
    with assert_warns(DeprecationWarning):
        res = np.fromstring(x_str, sep=',', count=5)
        assert_array_equal(res[:-1], x)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        res = np.fromstring(x_str, sep=',', count=4)
        assert_array_equal(res, x)