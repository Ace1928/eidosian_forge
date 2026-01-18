import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
@requires_memory(free_bytes=2 * 10000 * 1000 * 2)
def test_mean_overflow(self):
    a = masked_array(np.full((10000, 10000), 65535, dtype=np.uint16), mask=np.zeros((10000, 10000)))
    assert_equal(a.mean(), 65535.0)