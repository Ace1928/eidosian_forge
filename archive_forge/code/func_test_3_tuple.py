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
def test_3_tuple(self):
    for cls in (np.datetime64, np.timedelta64):
        self.assert_not_deprecated(cls, args=(1, ('ms', 2)))
        self.assert_not_deprecated(cls, args=(1, ('ms', 2, 1, None)))
        self.assert_deprecated(cls, args=(1, ('ms', 2, 'event')))
        self.assert_deprecated(cls, args=(1, ('ms', 2, 63)))
        self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 'event')))
        self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 63)))