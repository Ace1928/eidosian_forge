import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_min_int(self):
    a = np.array([np.iinfo(np.int_).min], dtype=np.int_)
    assert_allclose(a, a)