import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_with_view_float(self, float_frame):
    tf = np.round(float_frame).astype(np.int32)
    tf.astype(np.float32, copy=False)
    tf = float_frame.astype(np.float64)
    tf.astype(np.int64, copy=False)