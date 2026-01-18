import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_advange_large(self):
    rs = Generator(self.bit_generator(38219308213743))
    pcg = rs.bit_generator
    state = pcg.state
    initial_state = 287608843259529770491897792873167516365
    assert state['state']['state'] == initial_state
    pcg.advance(sum((2 ** i for i in (96, 64, 32, 16, 8, 4, 2, 1))))
    state = pcg.state['state']
    advanced_state = 277778083536782149546677086420637664879
    assert state['state'] == advanced_state