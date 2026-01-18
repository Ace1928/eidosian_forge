import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
@pytest.mark.parametrize('x', [1, -1.4, 0])
def test_callable_deprecate_kwarg(x):
    with tm.assert_produces_warning(FutureWarning):
        assert _f3(old=x) == _f3_mapping(x)