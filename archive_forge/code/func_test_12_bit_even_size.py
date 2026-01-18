import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_12_bit_even_size():
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-4ch-9S-12bit.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.int16))
        assert_equal(data.shape, (9, 4))
        assert_equal(data & 15, 0)
        assert_equal(data.max(), 32752)
        assert_equal(data[0, 0], 0)
        assert_equal(data.min(), -32768)
        del data