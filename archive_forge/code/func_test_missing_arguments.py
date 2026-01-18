import pytest
import numpy as np
from numpy.core._multiarray_tests import argparse_example_function as func
def test_missing_arguments():
    with pytest.raises(TypeError, match='missing required positional argument 0'):
        func()
    with pytest.raises(TypeError, match='missing required positional argument 0'):
        func(arg2=1, arg3=4)
    with pytest.raises(TypeError, match="missing required argument \\'arg2\\' \\(pos 1\\)"):
        func(1, arg3=5)