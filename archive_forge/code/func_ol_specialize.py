import unittest
from numba.tests.support import captured_stdout
from numba import typed
@overload(specialize)
def ol_specialize(x):
    iv = x.initial_value
    if iv is None:
        return lambda x: literally(x)
    assert iv == [1, 2, 3]
    return lambda x: x