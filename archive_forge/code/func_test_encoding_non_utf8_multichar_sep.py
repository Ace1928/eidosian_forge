from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('sep', ['::', '#####', '!!!', '123', '#1!c5', '%!c!d', '@@#4:2', '_!pd#_'])
@pytest.mark.parametrize('encoding', ['utf-16', 'utf-16-be', 'utf-16-le', 'utf-32', 'cp037'])
def test_encoding_non_utf8_multichar_sep(python_parser_only, sep, encoding):
    expected = DataFrame({'a': [1], 'b': [2]})
    parser = python_parser_only
    data = '1' + sep + '2'
    encoded_data = data.encode(encoding)
    result = parser.read_csv(BytesIO(encoded_data), sep=sep, names=['a', 'b'], encoding=encoding)
    tm.assert_frame_equal(result, expected)