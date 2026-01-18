import pytest
import numpy as np
from numpy.core._multiarray_tests import argparse_example_function as func
def test_invalid_integers():
    with pytest.raises(TypeError, match='integer argument expected, got float'):
        func(1.0)
    with pytest.raises(OverflowError):
        func(2 ** 100)