from types import SimpleNamespace
import pytest
from pandas.core.dtypes.common import is_float
import pandas._testing as tm
def test_assert_attr_equal(nulls_fixture):
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture
    tm.assert_attr_equal('na_value', obj, obj)