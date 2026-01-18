from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_default_separator(python_parser_only):
    data = 'aob\n1o2\n3o4'
    parser = python_parser_only
    expected = DataFrame({'a': [1, 3], 'b': [2, 4]})
    result = parser.read_csv(StringIO(data), sep=None)
    tm.assert_frame_equal(result, expected)