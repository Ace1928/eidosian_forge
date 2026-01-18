import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_prefix_conversions(self):
    smaller_units = ['M8[7000ms]', 'M8[2000us]', 'M8[1000ns]', 'M8[5000ns]', 'M8[2000ps]', 'M8[9000fs]', 'M8[1000as]', 'M8[2000000ps]', 'M8[1000000as]', 'M8[2000000000ps]', 'M8[1000000000as]']
    larger_units = ['M8[7s]', 'M8[2ms]', 'M8[us]', 'M8[5us]', 'M8[2ns]', 'M8[9ps]', 'M8[1fs]', 'M8[2us]', 'M8[1ps]', 'M8[2ms]', 'M8[1ns]']
    for larger_unit, smaller_unit in zip(larger_units, smaller_units):
        assert np.can_cast(larger_unit, smaller_unit, casting='safe')
        assert np.can_cast(smaller_unit, larger_unit, casting='safe')