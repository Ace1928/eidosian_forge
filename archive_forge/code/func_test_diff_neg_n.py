import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_neg_n(self, datetime_frame):
    rs = datetime_frame.diff(-1)
    xp = datetime_frame - datetime_frame.shift(-1)
    tm.assert_frame_equal(rs, xp)