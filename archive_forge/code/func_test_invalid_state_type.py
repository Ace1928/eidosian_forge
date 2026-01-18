import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_invalid_state_type(self):
    bit_generator = self.bit_generator(*self.data1['seed'])
    with pytest.raises(TypeError):
        bit_generator.state = {'1'}