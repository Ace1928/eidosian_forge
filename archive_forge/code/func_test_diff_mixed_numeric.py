import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_mixed_numeric(self, datetime_frame):
    tf = datetime_frame.astype('float32')
    the_diff = tf.diff(1)
    tm.assert_series_equal(the_diff['A'], tf['A'] - tf['A'].shift(1))