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
def test_field_access_by_title(self):
    s = 'Some long field name'
    if HAS_REFCOUNT:
        base = sys.getrefcount(s)
    t = np.dtype([((s, 'f1'), np.float64)])
    data = np.zeros(10, t)
    for i in range(10):
        str(data[['f1']])
        if HAS_REFCOUNT:
            assert_(base <= sys.getrefcount(s))