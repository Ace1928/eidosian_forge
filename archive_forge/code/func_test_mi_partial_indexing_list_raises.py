import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mi_partial_indexing_list_raises():
    frame = DataFrame(np.arange(12).reshape((4, 3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
    frame.index.names = ['key1', 'key2']
    frame.columns.names = ['state', 'color']
    with pytest.raises(KeyError, match='\\[2\\] not in index'):
        frame.loc[['b', 2], 'Colorado']