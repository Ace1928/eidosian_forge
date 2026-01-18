import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_fileslice_errors():
    arr = np.arange(24).reshape((2, 3, 4))
    fobj = BytesIO(arr.tobytes())
    _check_slicer((1,), arr, fobj, 0, 'C')
    with pytest.raises(ValueError):
        fileslice(fobj, (np.array([1]),), (2, 3, 4), arr.dtype)