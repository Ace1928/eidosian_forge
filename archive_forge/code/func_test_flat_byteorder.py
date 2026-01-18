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
def test_flat_byteorder(self):
    x = np.arange(10)
    assert_array_equal(x.astype('>i4'), x.astype('<i4').flat[:])
    assert_array_equal(x.astype('>i4').flat[:], x.astype('<i4'))