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
@pytest.mark.parametrize('type_code', np.typecodes['Complex'])
def test_complex_hashes(self, type_code):
    scalar = np.dtype(type_code).type
    for val in [np.pi + 1j, np.inf - 3j, 3j, 6.0 + 1j]:
        numpy_val = scalar(val)
        assert hash(complex(numpy_val)) == hash(numpy_val)