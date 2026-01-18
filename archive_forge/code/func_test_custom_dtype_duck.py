import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_custom_dtype_duck(self):

    class MyArray(list):

        @property
        def dtype(self):
            return complex
    a = MyArray([1 + 0j, 2 + 0j, 3 + 0j])
    assert_(iscomplexobj(a))