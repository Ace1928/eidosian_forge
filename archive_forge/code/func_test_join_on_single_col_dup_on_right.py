from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
@pytest.mark.parametrize('dtype', ['object', 'string[pyarrow]'])
def test_join_on_single_col_dup_on_right(left_no_dup, right_w_dups, dtype):
    if dtype == 'string[pyarrow]':
        pytest.importorskip('pyarrow')
    left_no_dup = left_no_dup.astype(dtype)
    right_w_dups.index = right_w_dups.index.astype(dtype)
    left_no_dup.join(right_w_dups, on='a', validate='one_to_many')
    msg = 'Merge keys are not unique in right dataset; not a one-to-one merge'
    with pytest.raises(MergeError, match=msg):
        left_no_dup.join(right_w_dups, on='a', validate='one_to_one')