import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_constructor_incorrect_parameter(self):
    with pytest.raises(TypeError, match='unexpected'):
        Result(x=1, y=2, z=3, w=123, beta=0.75, foo=999)