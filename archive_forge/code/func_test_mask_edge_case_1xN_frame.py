import numpy as np
from pandas import (
import pandas._testing as tm
def test_mask_edge_case_1xN_frame(self):
    df = DataFrame([[1, 2]])
    res = df.mask(DataFrame([[True, False]]))
    expec = DataFrame([[np.nan, 2]])
    tm.assert_frame_equal(res, expec)