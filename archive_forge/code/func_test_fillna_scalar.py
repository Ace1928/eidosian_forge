import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_fillna_scalar(self, data_missing):
    valid = data_missing[1]
    result = data_missing.fillna(valid)
    expected = data_missing.fillna(valid)
    tm.assert_extension_array_equal(result, expected)