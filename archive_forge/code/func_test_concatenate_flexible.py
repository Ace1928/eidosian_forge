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
def test_concatenate_flexible(self):
    data = masked_array(list(zip(np.random.rand(10), np.arange(10))), dtype=[('a', float), ('b', int)])
    test = concatenate([data[:5], data[5:]])
    assert_equal_records(test, data)