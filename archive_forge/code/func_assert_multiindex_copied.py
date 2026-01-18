from copy import (
import pytest
from pandas import MultiIndex
import pandas._testing as tm
def assert_multiindex_copied(copy, original):
    tm.assert_copy(copy.levels, original.levels)
    tm.assert_almost_equal(copy.codes, original.codes)
    tm.assert_almost_equal(copy.codes, original.codes)
    assert copy.codes is not original.codes
    assert copy.names == original.names
    assert copy.names is not original.names
    assert copy.sortorder == original.sortorder