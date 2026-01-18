import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def copyto_masked():
    arr = np.empty(3, dtype=dtype)
    np.copyto(arr, np.array([value, value, value]), casting='unsafe', where=[True, False, True])