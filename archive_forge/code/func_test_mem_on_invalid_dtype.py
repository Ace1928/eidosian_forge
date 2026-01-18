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
def test_mem_on_invalid_dtype(self):
    """Ticket #583"""
    assert_raises(ValueError, np.fromiter, [['12', ''], ['13', '']], str)