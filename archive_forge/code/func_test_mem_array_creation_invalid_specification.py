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
def test_mem_array_creation_invalid_specification(self):
    dt = np.dtype([('x', int), ('y', np.object_)])
    assert_raises(ValueError, np.array, [1, 'object'], dt)
    np.array([(1, 'object')], dt)