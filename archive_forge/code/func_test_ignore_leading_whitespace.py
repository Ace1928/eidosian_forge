from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
def test_ignore_leading_whitespace(all_parsers):
    parser = all_parsers
    data = ' a b c\n 1 2 3\n 4 5 6\n 7 8 9'
    if parser.engine == 'pyarrow':
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep='\\s+')
        return
    result = parser.read_csv(StringIO(data), sep='\\s+')
    expected = DataFrame({'a': [1, 4, 7], 'b': [2, 5, 8], 'c': [3, 6, 9]})
    tm.assert_frame_equal(result, expected)