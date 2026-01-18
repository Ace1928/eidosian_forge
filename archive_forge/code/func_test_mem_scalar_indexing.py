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
def test_mem_scalar_indexing(self):
    x = np.array([0], dtype=float)
    index = np.array(0, dtype=np.int32)
    x[index]