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
def test_complex_boolean_cast(self):
    for tp in [np.csingle, np.cdouble, np.clongdouble]:
        x = np.array([0, 0 + 0.5j, 0.5 + 0j], dtype=tp)
        assert_equal(x.astype(bool), np.array([0, 1, 1], dtype=bool))
        assert_(np.any(x))
        assert_(np.all(x[1:]))