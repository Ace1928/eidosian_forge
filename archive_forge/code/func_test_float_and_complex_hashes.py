import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('type_code', np.typecodes['AllFloat'])
def test_float_and_complex_hashes(self, type_code):
    scalar = np.dtype(type_code).type
    for val in [np.pi, np.inf, 3, 6.0]:
        numpy_val = scalar(val)
        if numpy_val.dtype.kind == 'c':
            val = complex(numpy_val)
        else:
            val = float(numpy_val)
        assert val == numpy_val
        assert hash(val) == hash(numpy_val)
    if hash(float(np.nan)) != hash(float(np.nan)):
        assert hash(scalar(np.nan)) != hash(scalar(np.nan))