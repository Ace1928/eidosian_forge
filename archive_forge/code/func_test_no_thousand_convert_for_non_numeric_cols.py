from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype,expected', [({'a': str, 'b': np.float64, 'c': np.int64}, DataFrame({'b': [16000.1, 0, 23000], 'c': [0, 4001, 131]})), (str, DataFrame({'b': ['16,000.1', '0', '23,000'], 'c': ['0', '4,001', '131']}))])
def test_no_thousand_convert_for_non_numeric_cols(python_parser_only, dtype, expected):
    parser = python_parser_only
    data = 'a;b;c\n0000,7995;16,000.1;0\n3,03,001,00514;0;4,001\n4923,600,041;23,000;131\n'
    result = parser.read_csv(StringIO(data), sep=';', dtype=dtype, thousands=',')
    expected.insert(0, 'a', ['0000,7995', '3,03,001,00514', '4923,600,041'])
    tm.assert_frame_equal(result, expected)