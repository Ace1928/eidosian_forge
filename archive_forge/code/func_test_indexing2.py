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
def test_indexing2(self):
    descr = [('x', 'i4', (2,))]
    buffer = ([3, 2],)
    h = np.array(buffer, dtype=descr)
    h['x']