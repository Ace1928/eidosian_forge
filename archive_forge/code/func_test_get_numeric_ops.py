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
def test_get_numeric_ops(self):
    from numpy.core._multiarray_tests import getset_numericops
    self.assert_deprecated(getset_numericops, num=2)
    self.assert_deprecated(np.set_numeric_ops, kwargs={})
    assert_raises(ValueError, np.set_numeric_ops, add='abc')