import pytest
from pandas.util._decorators import deprecate_kwarg
import pandas._testing as tm
def test_bad_deprecate_kwarg():
    msg = 'mapping from old to new argument values must be dict or callable!'
    with pytest.raises(TypeError, match=msg):

        @deprecate_kwarg('old', 'new', 0)
        def f4(new=None):
            return new