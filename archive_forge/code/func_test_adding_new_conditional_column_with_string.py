from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize(('dtype', 'infer_string'), [(object, False), ('string[pyarrow_numpy]', True)])
def test_adding_new_conditional_column_with_string(dtype, infer_string) -> None:
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, 2], 'b': [3, 4]})
    with pd.option_context('future.infer_string', infer_string):
        df.loc[df['a'] == 1, 'c'] = '1'
    expected = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': ['1', float('nan')]}).astype({'a': 'int64', 'b': 'int64', 'c': dtype})
    tm.assert_frame_equal(df, expected)