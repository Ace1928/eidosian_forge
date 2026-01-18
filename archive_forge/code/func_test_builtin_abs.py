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
@pytest.mark.parametrize('dtype', floating_types + complex_floating_types)
def test_builtin_abs(self, dtype):
    if sys.platform == 'cygwin' and dtype == np.clongdouble and (_pep440.parse(platform.release().split('-')[0]) < _pep440.Version('3.3.0')):
        pytest.xfail(reason='absl is computed in double precision on cygwin < 3.3')
    self._test_abs_func(abs, dtype)