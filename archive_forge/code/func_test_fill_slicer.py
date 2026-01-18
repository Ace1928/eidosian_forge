import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_fill_slicer():
    assert fill_slicer(slice(None), 10) == slice(0, 10, 1)
    assert fill_slicer(slice(11), 10) == slice(0, 10, 1)
    assert fill_slicer(slice(1, 11), 10) == slice(1, 10, 1)
    assert fill_slicer(slice(1, 1), 10) == slice(1, 1, 1)
    assert fill_slicer(slice(1, 11, 2), 10) == slice(1, 10, 2)
    assert fill_slicer(slice(0, 11, 3), 10) == slice(0, 10, 3)
    assert fill_slicer(slice(1, 11, 3), 10) == slice(1, 10, 3)
    assert fill_slicer(slice(None, None, -1), 10) == slice(9, None, -1)
    assert fill_slicer(slice(11, None, -1), 10) == slice(9, None, -1)
    assert fill_slicer(slice(None, 1, -1), 10) == slice(9, 1, -1)
    assert fill_slicer(slice(None, None, -2), 10) == slice(9, None, -2)
    assert fill_slicer(slice(None, None, -3), 10) == slice(9, None, -3)
    assert fill_slicer(slice(None, 0, -3), 10) == slice(9, 0, -3)
    assert fill_slicer(slice(None, -4, -1), 10) == slice(9, 6, -1)
    assert fill_slicer(slice(-4, -2, 1), 10) == slice(6, 8, 1)
    assert fill_slicer(slice(3, 2, 1), 10) == slice(3, 2, 1)
    assert fill_slicer(slice(2, 3, -1), 10) == slice(2, 3, -1)