import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
@pytest.mark.parametrize('wt', ['foobar', 1])
def test_invalid_constructor_wintype(frame_or_series, wt):
    pytest.importorskip('scipy')
    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match='Invalid win_type'):
        c(win_type=wt, window=2)