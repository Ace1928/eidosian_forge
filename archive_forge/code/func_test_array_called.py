import pytest
import warnings
import numpy as np
def test_array_called():

    class Wrapper:
        val = '0' * 100

        def __array__(self, result=None):
            return np.array([self.val], dtype=object)
    wrapped = Wrapper()
    arr = np.array(wrapped, dtype=str)
    assert arr.dtype == 'U100'
    assert arr[0] == Wrapper.val