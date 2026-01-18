import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_64_bit_even_size():
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-3ch-5S-64bit.wav'
        rate, data = wavfile.read(datafile(filename), mmap=False)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.int64))
        assert_equal(data.shape, (5, 3))
        correct = [[-9223372036854775808, -9223372036854775807, -2], [-4611686018427387904, -4611686018427387903, -1], [+0, +0, +0], [+4611686018427387904, +4611686018427387903, +1], [+9223372036854775807, +9223372036854775807, +2]]
        assert_equal(data, correct)
        del data