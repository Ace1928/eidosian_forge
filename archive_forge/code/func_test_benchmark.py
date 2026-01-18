import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_benchmark(self):
    bit_generator = self.bit_generator(*self.data1['seed'])
    bit_generator._benchmark(1)
    bit_generator._benchmark(1, 'double')
    with pytest.raises(ValueError):
        bit_generator._benchmark(1, 'int32')