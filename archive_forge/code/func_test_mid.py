import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_mid(self):
    assert_array_equal(waveforms.unit_impulse((3, 3), 'mid'), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_array_equal(waveforms.unit_impulse(9, 'mid'), [0, 0, 0, 0, 1, 0, 0, 0, 0])