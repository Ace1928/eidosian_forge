import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_single(self, df):
    result = crosstab(df['A'], df['C'])
    expected = df.groupby(['A', 'C']).size().unstack()
    tm.assert_frame_equal(result, expected.fillna(0).astype(np.int64))