import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def assert_startswith(a, b):
    assert_equal(a[:len(b)], b)