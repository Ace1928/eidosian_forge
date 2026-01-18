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
def test_masked_where_oddities(self):
    atest = ones((10, 10, 10), dtype=float)
    btest = zeros(atest.shape, MaskType)
    ctest = masked_where(btest, atest)
    assert_equal(atest, ctest)