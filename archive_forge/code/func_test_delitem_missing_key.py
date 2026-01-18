import pytest
from pandas import (
import pandas._testing as tm
def test_delitem_missing_key(self):
    s = Series(dtype=object)
    with pytest.raises(KeyError, match='^0$'):
        del s[0]