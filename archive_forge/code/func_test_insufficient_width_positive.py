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
def test_insufficient_width_positive(self):
    args = (10,)
    kwargs = {'width': 2}
    self.message = 'Insufficient bit width provided. This behavior will raise an error in the future.'
    self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)