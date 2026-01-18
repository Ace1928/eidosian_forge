from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_empty_except_index(engine):
    expected = DataFrame(index=['a'])
    result = expected.apply(lambda x: x['a'], axis=1, engine=engine)
    tm.assert_frame_equal(result, expected)