import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_no_args_raises(self):
    gr = Series([1, 2]).groupby([0, 1])
    with pytest.raises(TypeError, match='Must provide'):
        gr.agg()
    result = gr.agg([])
    expected = DataFrame(columns=[])
    tm.assert_frame_equal(result, expected)