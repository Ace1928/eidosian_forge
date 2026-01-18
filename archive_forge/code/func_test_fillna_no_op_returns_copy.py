import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_fillna_no_op_returns_copy(self, data):
    data = data[~data.isna()]
    valid = data[0]
    result = data.fillna(valid)
    assert result is not data
    tm.assert_extension_array_equal(result, data)
    result = data._pad_or_backfill(method='backfill')
    assert result is not data
    tm.assert_extension_array_equal(result, data)