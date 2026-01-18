import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_fail_to_match_resource_warning():
    category = ResourceWarning
    match = '\\d+'
    unmatched = "Did not see warning 'ResourceWarning' matching '\\\\d\\+'. The emitted warning messages are \\[ResourceWarning\\('This is not a match.'\\), ResourceWarning\\('Another unmatched warning.'\\)\\]"
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn('This is not a match.', category)
            warnings.warn('Another unmatched warning.', category)