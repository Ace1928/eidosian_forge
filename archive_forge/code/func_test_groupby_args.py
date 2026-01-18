from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_args(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    msg = "You have to supply one of 'by' and 'level'"
    with pytest.raises(TypeError, match=msg):
        frame.groupby()
    msg = "You have to supply one of 'by' and 'level'"
    with pytest.raises(TypeError, match=msg):
        frame.groupby(by=None, level=None)