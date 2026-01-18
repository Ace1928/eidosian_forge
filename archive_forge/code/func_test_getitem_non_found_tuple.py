import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_non_found_tuple():
    df = DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd']).set_index(['a', 'b', 'c'])
    with pytest.raises(KeyError, match='\\(2\\.0, 2\\.0, 3\\.0\\)'):
        df.loc[2.0, 2.0, 3.0]