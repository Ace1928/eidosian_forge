import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
@pytest.mark.parametrize('t', numeric_types)
def test_names_are_undersood_by_dtype(self, t):
    """ Test the dtype constructor maps names back to the type """
    assert np.dtype(t.__name__).type is t