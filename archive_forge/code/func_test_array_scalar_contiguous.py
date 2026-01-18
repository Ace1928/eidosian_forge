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
def test_array_scalar_contiguous(self):
    assert_(np.array(1.0).flags.c_contiguous)
    assert_(np.array(1.0).flags.f_contiguous)
    assert_(np.array(np.float32(1.0)).flags.c_contiguous)
    assert_(np.array(np.float32(1.0)).flags.f_contiguous)