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
def test_assert_deprecated(self):
    test_case_instance = _DeprecationTestCase()
    test_case_instance.setup_method()
    assert_raises(AssertionError, test_case_instance.assert_deprecated, lambda: None)

    def foo():
        warnings.warn('foo', category=DeprecationWarning, stacklevel=2)
    test_case_instance.assert_deprecated(foo)
    test_case_instance.teardown_method()