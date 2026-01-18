import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_fail_to_match_future_warning():
    category = FutureWarning
    match = 'Warning'
    unmatched = "Did not see warning 'FutureWarning' matching 'Warning'. The emitted warning messages are \\[FutureWarning\\('This is not a match.'\\), FutureWarning\\('Another unmatched warning.'\\)\\]"
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn('This is not a match.', category)
            warnings.warn('Another unmatched warning.', category)