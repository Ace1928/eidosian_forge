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
def test_hardmask_again(self):
    d = arange(5)
    n = [0, 0, 0, 1, 1]
    m = make_mask(n)
    xh = array(d, mask=m, hard_mask=True)
    xh[4:5] = 999
    xh[0:1] = 999
    assert_equal(xh._data, [999, 1, 2, 3, 4])