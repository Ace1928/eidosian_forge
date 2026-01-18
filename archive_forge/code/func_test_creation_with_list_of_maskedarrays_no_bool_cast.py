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
def test_creation_with_list_of_maskedarrays_no_bool_cast(self):
    masked_str = np.ma.masked_array(['a', 'b'], mask=[True, False])
    normal_int = np.arange(2)
    res = np.ma.asarray([masked_str, normal_int], dtype='U21')
    assert_array_equal(res.mask, [[True, False], [False, False]])

    class NotBool:

        def __bool__(self):
            raise ValueError('not a bool!')
    masked_obj = np.ma.masked_array([NotBool(), 'b'], mask=[True, False])
    with pytest.raises(ValueError, match='not a bool!'):
        np.asarray([masked_obj], dtype=bool)
    res = np.ma.asarray([masked_obj, normal_int])
    assert_array_equal(res.mask, [[True, False], [False, False]])