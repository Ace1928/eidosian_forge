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
def test_endian_recarray(self):
    dt = np.dtype([('head', '>u4'), ('data', '>u4', 2)])
    buf = np.recarray(1, dtype=dt)
    buf[0]['head'] = 1
    buf[0]['data'][:] = [1, 1]
    h = buf[0]['head']
    d = buf[0]['data'][0]
    buf[0]['head'] = h
    buf[0]['data'][0] = d
    assert_(buf[0]['head'] == 1)