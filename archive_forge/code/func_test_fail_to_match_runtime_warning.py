import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_fail_to_match_runtime_warning():
    category = RuntimeWarning
    match = 'Did not see this warning'
    unmatched = "Did not see warning 'RuntimeWarning' matching 'Did not see this warning'. The emitted warning messages are \\[RuntimeWarning\\('This is not a match.'\\), RuntimeWarning\\('Another unmatched warning.'\\)\\]"
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn('This is not a match.', category)
            warnings.warn('Another unmatched warning.', category)