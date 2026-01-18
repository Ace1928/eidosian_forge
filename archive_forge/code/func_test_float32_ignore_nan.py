import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_float32_ignore_nan(self):
    offset = np.uint32(65535)
    nan1_i32 = np.array(np.nan, dtype=np.float32).view(np.uint32)
    nan2_i32 = nan1_i32 ^ offset
    nan1_f32 = nan1_i32.view(np.float32)
    nan2_f32 = nan2_i32.view(np.float32)
    assert_array_max_ulp(nan1_f32, nan2_f32, 0)