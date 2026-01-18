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
@pytest.mark.parametrize('type_code', np.typecodes['AllInteger'])
def test_integer_hashes(self, type_code):
    scalar = np.dtype(type_code).type
    for i in range(128):
        assert hash(i) == hash(scalar(i))