import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:f1:FutureWarning')
def test_assert_produces_warning_honors_filter():
    msg = 'Caused unexpected warning\\(s\\)'
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            f()
    with tm.assert_produces_warning(RuntimeWarning, raise_on_extra_warnings=False):
        f()