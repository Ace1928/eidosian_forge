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
def test_lexsort_zerolen_custom_strides_2d(self):
    xs = np.array([], dtype='i8')
    xs.shape = (0, 2)
    xs.strides = (16, 16)
    assert np.lexsort((xs,), axis=0).shape[0] == 0
    xs.shape = (2, 0)
    xs.strides = (16, 16)
    assert np.lexsort((xs,), axis=0).shape[0] == 2