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
def test_lexsort(self):
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_equal(np.lexsort(v), 0)