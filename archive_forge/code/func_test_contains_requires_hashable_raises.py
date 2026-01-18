import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_contains_requires_hashable_raises(self, index):
    if isinstance(index, MultiIndex):
        return
    msg = "unhashable type: 'list'"
    with pytest.raises(TypeError, match=msg):
        [] in index
    msg = '|'.join(["unhashable type: 'dict'", 'must be real number, not dict', 'an integer is required', '\\{\\}', "pandas\\._libs\\.interval\\.IntervalTree' is not iterable"])
    with pytest.raises(TypeError, match=msg):
        {} in index._engine