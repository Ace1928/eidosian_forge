import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def boolean_array_assignment():
    arr = np.empty(3, dtype=dtype)
    arr[[True, False, True]] = np.array([value, value])