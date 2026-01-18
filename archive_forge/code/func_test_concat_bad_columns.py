import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_concat_bad_columns(styler):
    msg = '`other.data` must have same columns as `Styler.data'
    with pytest.raises(ValueError, match=msg):
        styler.concat(DataFrame([[1, 2]]).style)