import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
@pytest.mark.leaks_references(reason='dynamically creates custom dtype.')
def test_custom_structured_dtype(self):

    class mytype:
        pass
    blueprint = np.dtype([('field', object)])
    dt = create_custom_field_dtype(blueprint, mytype, 0)
    assert dt.type == mytype
    assert np.dtype(mytype) == np.dtype('O')