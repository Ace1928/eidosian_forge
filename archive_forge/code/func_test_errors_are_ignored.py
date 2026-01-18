import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.skipif(sys.flags.optimize == 2, reason='Python running -OO')
def test_errors_are_ignored(self):
    prev_doc = np.core.flatiter.index.__doc__
    np.add_newdoc('numpy.core', 'flatiter', ('index', 'bad docstring'))
    assert prev_doc == np.core.flatiter.index.__doc__