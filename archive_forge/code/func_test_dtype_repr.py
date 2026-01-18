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
def test_dtype_repr(self):
    dt1 = np.dtype(('uint32', 2))
    dt2 = np.dtype(('uint32', (2,)))
    assert_equal(dt1.__repr__(), dt2.__repr__())