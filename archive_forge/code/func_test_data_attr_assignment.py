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
def test_data_attr_assignment(self):
    a = np.arange(10)
    b = np.linspace(0, 1, 10)
    self.message = "Assigning the 'data' attribute is an inherently unsafe operation and will be removed in the future."
    self.assert_deprecated(a.__setattr__, args=('data', b.data))