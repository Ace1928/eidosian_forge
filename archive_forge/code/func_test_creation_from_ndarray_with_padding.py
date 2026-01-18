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
def test_creation_from_ndarray_with_padding(self):
    x = np.array([('A', 0)], dtype={'names': ['f0', 'f1'], 'formats': ['S4', 'i8'], 'offsets': [0, 8]})
    array(x)