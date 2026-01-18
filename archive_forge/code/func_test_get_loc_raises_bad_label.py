import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_loc_raises_bad_label(self):
    index = Index([0, 1, 2])
    with pytest.raises(InvalidIndexError, match='\\[1, 2\\]'):
        index.get_loc([1, 2])