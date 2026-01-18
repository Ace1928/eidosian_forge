from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('attribute', ['__array_interface__', '__array__', '__array_struct__'])
@pytest.mark.parametrize('error', [RecursionError, MemoryError])
def test_bad_array_like_attributes(self, attribute, error):

    class BadInterface:

        def __getattr__(self, attr):
            if attr == attribute:
                raise error
            super().__getattr__(attr)
    with pytest.raises(error):
        np.array(BadInterface())