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
def test_array_construction(self):
    for scalar_type in self.deprecated_types:
        self.assert_deprecated(np.array, args=([], scalar_type))