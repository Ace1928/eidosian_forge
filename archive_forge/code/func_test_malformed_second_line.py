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
def test_malformed_second_line(all_parsers):
    parser = all_parsers
    data = '\na\nb\n'
    result = parser.read_csv(StringIO(data), skip_blank_lines=False, header=1)
    expected = DataFrame({'a': ['b']})
    tm.assert_frame_equal(result, expected)