import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def slicer_samples(shape):
    """Generator returns slice samples for given `shape`"""
    ndim = len(shape)
    slicers_list = []
    for i in range(ndim):
        slicers_list.append(_slices_for_len(shape[i]))
        yield from product(*slicers_list)
    yield (None,)
    if ndim == 0:
        return
    yield (None, 0)
    yield (None, np.array(0))
    yield (0, None)
    yield (np.array(0), None)
    yield (Ellipsis, -1)
    yield (Ellipsis, np.array(-1))
    yield (-1, Ellipsis)
    yield (np.array(-1), Ellipsis)
    yield (None, Ellipsis)
    yield (Ellipsis, None)
    yield (Ellipsis, None, None)
    if ndim == 1:
        return
    yield (0, None, slice(None))
    yield (np.array(0), None, slice(None))
    yield (Ellipsis, -1, None)
    yield (Ellipsis, np.array(-1), None)
    yield (0, Ellipsis, None)
    yield (np.array(0), Ellipsis, None)
    if ndim == 2:
        return
    yield (slice(None), 0, -1, None)
    yield (slice(None), np.array(0), np.array(-1), None)
    yield (np.array(0), slice(None), np.array(-1), None)