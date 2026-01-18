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
def test_misaligned_dot_product_objects(self):
    a = np.array([[(1, 'a'), (0, 'a')], [(0, 'a'), (1, 'a')]], dtype='O,c')
    b = np.array([[(4, 'a'), (1, 'a')], [(2, 'a'), (2, 'a')]], dtype='O,c')
    np.dot(a['f0'], b['f0'])