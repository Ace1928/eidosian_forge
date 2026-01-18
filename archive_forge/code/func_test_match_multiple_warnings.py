import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_match_multiple_warnings():
    category = (FutureWarning, UserWarning)
    with tm.assert_produces_warning(category, match='^Match this'):
        warnings.warn('Match this', FutureWarning)
        warnings.warn('Match this too', UserWarning)