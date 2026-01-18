import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_bad_header_uniform_error(all_parsers):
    parser = all_parsers
    data = '+++123456789...\ncol1,col2,col3,col4\n1,2,3,4\n'
    msg = 'Expected 2 fields in line 2, saw 4'
    if parser.engine == 'c':
        msg = 'Could not construct index. Requested to use 1 number of columns, but 3 left to parse.'
    elif parser.engine == 'pyarrow':
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), index_col=0, on_bad_lines='error')