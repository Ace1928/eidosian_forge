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
def test_lexsort_zerolen_custom_strides(self):
    xs = np.array([], dtype='i8')
    assert np.lexsort((xs,)).shape[0] == 0
    xs.strides = (16,)
    assert np.lexsort((xs,)).shape[0] == 0