from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_index_col_false_and_header_none(python_parser_only):
    parser = python_parser_only
    data = '\n0.5,0.03\n0.1,0.2,0.3,2\n'
    result = parser.read_csv_check_warnings(ParserWarning, 'Length of header', StringIO(data), sep=',', header=None, index_col=False)
    expected = DataFrame({0: [0.5, 0.1], 1: [0.03, 0.2]})
    tm.assert_frame_equal(result, expected)