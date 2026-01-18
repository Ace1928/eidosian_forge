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
def test_fillvalue_bytes_or_str(self):
    a = empty(shape=(3,), dtype='(2)3S,(2)3U')
    assert_equal(a['f0'].fill_value, default_fill_value(b'spam'))
    assert_equal(a['f1'].fill_value, default_fill_value('eggs'))