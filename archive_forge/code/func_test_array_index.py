import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_array_index(self):
    a = np.array([1, 2, 3])
    a2 = np.array([[1, 2, 3]])
    assert_equal(a[np.where(a == 3)], a2[np.where(a2 == 3)])