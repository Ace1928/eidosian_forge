import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_at_least_one_field_name_required(self):
    with pytest.raises(ValueError, match='at least one name'):
        _make_tuple_bunch('Qwerty', [], ['a', 'b'])