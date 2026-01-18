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
@skip_pyarrow
def test_first_row_bom(all_parsers):
    parser = all_parsers
    data = '\ufeff"Head1"\t"Head2"\t"Head3"'
    result = parser.read_csv(StringIO(data), delimiter='\t')
    expected = DataFrame(columns=['Head1', 'Head2', 'Head3'])
    tm.assert_frame_equal(result, expected)