import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_multi_index_header(self):
    columns = MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
    df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
    df.columns = columns
    header = ['a', 'b', 'c', 'd']
    result = df.to_csv(header=header)
    expected_rows = [',a,b,c,d', '0,1,2,3,4', '1,5,6,7,8']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert result == expected