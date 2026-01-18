import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def flat_assignment():
    arr = np.empty((3,), dtype=dtype)
    values = np.array([value, value, value])
    arr.flat[:] = values