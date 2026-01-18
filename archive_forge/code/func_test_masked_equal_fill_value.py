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
def test_masked_equal_fill_value(self):
    x = [1, 2, 3]
    mx = masked_equal(x, 3)
    assert_equal(mx._mask, [0, 0, 1])
    assert_equal(mx.fill_value, 3)