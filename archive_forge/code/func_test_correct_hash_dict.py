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
def test_correct_hash_dict(self):
    all_types = set(np.sctypeDict.values()) - {np.void}
    for t in all_types:
        val = t()
        try:
            hash(val)
        except TypeError as e:
            assert_equal(t.__hash__, None)
        else:
            assert_(t.__hash__ != None)