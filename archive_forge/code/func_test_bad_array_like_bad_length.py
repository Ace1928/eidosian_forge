from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
@pytest.mark.parametrize('error', [RecursionError, MemoryError])
def test_bad_array_like_bad_length(self, error):

    class BadSequence:

        def __len__(self):
            raise error

        def __getitem__(self):
            return 1
    with pytest.raises(error):
        np.array(BadSequence())