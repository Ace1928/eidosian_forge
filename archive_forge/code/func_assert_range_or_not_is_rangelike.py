from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def assert_range_or_not_is_rangelike(index):
    """
    Check that we either have a RangeIndex or that this index *cannot*
    be represented as a RangeIndex.
    """
    if not isinstance(index, RangeIndex) and len(index) > 0:
        diff = index[:-1] - index[1:]
        assert not (diff == diff[0]).all()