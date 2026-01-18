from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
def test_invalid_dtype_backend(all_parsers):
    parser = all_parsers
    msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        parser.read_csv('test', dtype_backend='numpy')