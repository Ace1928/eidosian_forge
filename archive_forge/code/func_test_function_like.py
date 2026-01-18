import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_function_like():
    assert type(np.mean) is np.core._multiarray_umath._ArrayFunctionDispatcher

    class MyClass:

        def __array__(self):
            return np.arange(3)
        func1 = staticmethod(np.mean)
        func2 = np.mean
        func3 = classmethod(np.mean)
    m = MyClass()
    assert m.func1([10]) == 10
    assert m.func2() == 1
    with pytest.raises(TypeError, match='unsupported operand type'):
        m.func3()
    bound = np.mean.__get__(m, MyClass)
    assert bound() == 1
    bound = np.mean.__get__(None, MyClass)
    assert bound([10]) == 10
    bound = np.mean.__get__(MyClass)
    with pytest.raises(TypeError, match='unsupported operand type'):
        bound()