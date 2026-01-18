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
def test_fillvalue_datetime_timedelta(self):
    for timecode in ('as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y'):
        control = numpy.datetime64('NaT', timecode)
        test = default_fill_value(numpy.dtype('<M8[' + timecode + ']'))
        np.testing.assert_equal(test, control)
        control = numpy.timedelta64('NaT', timecode)
        test = default_fill_value(numpy.dtype('<m8[' + timecode + ']'))
        np.testing.assert_equal(test, control)