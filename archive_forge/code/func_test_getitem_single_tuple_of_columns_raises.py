from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_getitem_single_tuple_of_columns_raises(self, df):
    with pytest.raises(ValueError, match='Cannot subset columns with a tuple'):
        df.groupby('A')['C', 'D'].mean()