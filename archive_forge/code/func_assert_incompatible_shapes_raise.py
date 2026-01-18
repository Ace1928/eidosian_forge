import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def assert_incompatible_shapes_raise(input_shapes):
    inarrays = [np.zeros(s) for s in input_shapes]
    assert_raises(ValueError, broadcast_arrays, *inarrays)