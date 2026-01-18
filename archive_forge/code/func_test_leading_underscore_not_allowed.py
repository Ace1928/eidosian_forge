import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
@pytest.mark.parametrize('args', [('Foo', ['_a'], ['x']), ('Foo', ['a'], ['_x'])])
def test_leading_underscore_not_allowed(self, args):
    with pytest.raises(ValueError, match='underscore'):
        _make_tuple_bunch(*args)