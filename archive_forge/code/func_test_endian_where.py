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
def test_endian_where(self):
    net = np.zeros(3, dtype='>f4')
    net[1] = 0.00458849
    net[2] = 0.605202
    max_net = net.max()
    test = np.where(net <= 0.0, max_net, net)
    correct = np.array([0.60520202, 0.00458849, 0.60520202])
    assert_array_almost_equal(test, correct)