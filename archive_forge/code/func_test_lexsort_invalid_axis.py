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
def test_lexsort_invalid_axis(self):
    assert_raises(np.AxisError, np.lexsort, (np.arange(1),), axis=2)
    assert_raises(np.AxisError, np.lexsort, (np.array([]),), axis=1)
    assert_raises(np.AxisError, np.lexsort, (np.array(1),), axis=10)