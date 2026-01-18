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
def test_1000_sep(all_parsers):
    parser = all_parsers
    data = 'A|B|C\n1|2,334|5\n10|13|10.\n'
    expected = DataFrame({'A': [1, 10], 'B': [2334, 13], 'C': [5, 10.0]})
    if parser.engine == 'pyarrow':
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep='|', thousands=',')
        return
    result = parser.read_csv(StringIO(data), sep='|', thousands=',')
    tm.assert_frame_equal(result, expected)