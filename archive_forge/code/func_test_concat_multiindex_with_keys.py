from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiindex_with_keys(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    index = frame.index
    result = concat([frame, frame], keys=[0, 1], names=['iteration'])
    assert result.index.names == ('iteration',) + index.names
    tm.assert_frame_equal(result.loc[0], frame)
    tm.assert_frame_equal(result.loc[1], frame)
    assert result.index.nlevels == 3