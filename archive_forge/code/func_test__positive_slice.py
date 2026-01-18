import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test__positive_slice():
    assert _positive_slice(slice(0, 5, 1)) == slice(0, 5, 1)
    assert _positive_slice(slice(1, 5, 3)) == slice(1, 5, 3)
    assert _positive_slice(slice(4, None, -2)) == slice(0, 5, 2)
    assert _positive_slice(slice(4, None, -1)) == slice(0, 5, 1)
    assert _positive_slice(slice(4, 1, -1)) == slice(2, 5, 1)
    assert _positive_slice(slice(4, 1, -2)) == slice(2, 5, 2)