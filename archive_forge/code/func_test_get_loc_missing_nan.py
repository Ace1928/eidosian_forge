from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_missing_nan(self):
    idx = MultiIndex.from_arrays([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(idx.get_loc(1), slice)
    with pytest.raises(KeyError, match='^3$'):
        idx.get_loc(3)
    with pytest.raises(KeyError, match='^nan$'):
        idx.get_loc(np.nan)
    with pytest.raises(InvalidIndexError, match='\\[nan\\]'):
        idx.get_loc([np.nan])