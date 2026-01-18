import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_53_bit_odd_size():
    filename = 'test-8000Hz-le-3ch-5S-53bit.wav'
    rate, data = wavfile.read(datafile(filename), mmap=False)
    assert_equal(rate, 8000)
    assert_(np.issubdtype(data.dtype, np.int64))
    assert_equal(data.shape, (5, 3))
    assert_equal(data & 2047, 0)
    correct = [[-9223372036854775808, -9223372036854773760, -4096], [-4611686018427387904, -4611686018427385856, -2048], [+0, +0, +0], [+4611686018427387904, +4611686018427385856, +2048], [+9223372036854773760, +9223372036854773760, +4096]]
    assert_equal(data, correct)