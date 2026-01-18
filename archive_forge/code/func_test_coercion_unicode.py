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
@pytest.mark.xfail(reason='See gh-9750')
def test_coercion_unicode(self):
    a_u = np.zeros((), 'U10')
    a_u[()] = np.ma.masked
    assert_equal(a_u[()], '--')