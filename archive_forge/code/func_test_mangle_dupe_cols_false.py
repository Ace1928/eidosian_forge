from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
def test_mangle_dupe_cols_false(self):
    data = 'a b c\n1 2 3'
    for engine in ('c', 'python'):
        with pytest.raises(TypeError, match='unexpected keyword'):
            read_csv(StringIO(data), engine=engine, mangle_dupe_cols=True)