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
def test_duplicate_field_names_assign(self):
    ra = np.fromiter(((i * 3, i * 2) for i in range(10)), dtype='i8,f8')
    ra.dtype.names = ('f1', 'f2')
    repr(ra)
    assert_raises(ValueError, setattr, ra.dtype, 'names', ('f1', 'f1'))