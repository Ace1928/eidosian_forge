import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_object_errors(self):

    class CustomException(Exception):
        pass

    class DestructoBox:

        def __init__(self, value, destruct):
            self._val = value
            self._destruct = destruct

        def __add__(self, other):
            tmp = self._val + other._val
            if tmp >= self._destruct:
                raise CustomException
            else:
                self._val = tmp
                return self

        def __radd__(self, other):
            if other == 0:
                return self
            else:
                return self.__add__(other)

        def __mul__(self, other):
            tmp = self._val * other._val
            if tmp >= self._destruct:
                raise CustomException
            else:
                self._val = tmp
                return self

        def __rmul__(self, other):
            if other == 0:
                return self
            else:
                return self.__mul__(other)
    a = np.array([DestructoBox(i, 5) for i in range(1, 10)], dtype='object').reshape(3, 3)
    assert_raises(CustomException, np.einsum, 'ij->i', a)
    b = np.array([DestructoBox(i, 100) for i in range(0, 27)], dtype='object').reshape(3, 3, 3)
    assert_raises(CustomException, np.einsum, 'i...k->...', b)
    b = np.array([DestructoBox(i, 55) for i in range(1, 4)], dtype='object')
    assert_raises(CustomException, np.einsum, 'ij, j', a, b)
    assert_raises(CustomException, np.einsum, 'ij, jh', a, a)
    assert_raises(CustomException, np.einsum, 'ij->', a)