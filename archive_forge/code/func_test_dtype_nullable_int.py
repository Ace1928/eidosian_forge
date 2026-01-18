from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def test_dtype_nullable_int(parser):
    df_result = read_xml(StringIO(xml_types), dtype={'sides': 'Int64'}, parser=parser)
    df_iter = read_xml_iterparse(xml_types, parser=parser, dtype={'sides': 'Int64'}, iterparse={'row': ['shape', 'degrees', 'sides']})
    df_expected = DataFrame({'shape': ['square', 'circle', 'triangle'], 'degrees': [360, 360, 180], 'sides': Series([4.0, float('nan'), 3.0]).astype('Int64')})
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)