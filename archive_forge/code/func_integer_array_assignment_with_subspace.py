import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def integer_array_assignment_with_subspace():
    arr = np.empty((5, 3), dtype=dtype)
    values = np.array([value, value, value])
    arr[[0, 2]] = values