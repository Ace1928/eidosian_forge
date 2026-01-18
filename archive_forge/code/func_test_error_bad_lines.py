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
def test_error_bad_lines(all_parsers):
    parser = all_parsers
    data = 'a\n1\n1,2,3\n4\n5,6,7'
    msg = 'Expected 1 fields in line 3, saw 3'
    if parser.engine == 'pyarrow':
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), on_bad_lines='error')