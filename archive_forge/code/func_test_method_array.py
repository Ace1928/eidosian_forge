import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_method_array(self):
    r = np.rec.array(b'abcdefg' * 100, formats='i2,a3,i4', shape=3, byteorder='big')
    assert_equal(r[1].item(), (25444, b'efg', 1633837924))