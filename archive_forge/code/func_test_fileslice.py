import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_fileslice():
    shapes = (15, 16, 17)
    for n_dim in range(1, len(shapes) + 1):
        shape = shapes[:n_dim]
        arr = np.arange(np.prod(shape)).reshape(shape)
        for order in 'FC':
            for offset in (0, 20):
                fobj = BytesIO()
                fobj.write(b'\x00' * offset)
                fobj.write(arr.tobytes(order=order))
                for sliceobj in slicer_samples(shape):
                    _check_slicer(sliceobj, arr, fobj, offset, order)