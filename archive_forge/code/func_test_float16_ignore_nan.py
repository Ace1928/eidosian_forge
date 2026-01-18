import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_float16_ignore_nan(self):
    offset = np.uint16(255)
    nan1_i16 = np.array(np.nan, dtype=np.float16).view(np.uint16)
    nan2_i16 = nan1_i16 ^ offset
    nan1_f16 = nan1_i16.view(np.float16)
    nan2_f16 = nan2_i16.view(np.float16)
    assert_array_max_ulp(nan1_f16, nan2_f16, 0)