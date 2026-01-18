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
def test_junk_in_string_fields_of_recarray(self):
    r = np.array([[b'abc']], dtype=[('var1', '|S20')])
    assert_(asbytes(r['var1'][0][0]) == b'abc')