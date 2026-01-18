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
def test_dict_keys_as_names(all_parsers):
    data = '1,2'
    keys = {'a': int, 'b': int}.keys()
    parser = all_parsers
    result = parser.read_csv(StringIO(data), names=keys)
    expected = DataFrame({'a': [1], 'b': [2]})
    tm.assert_frame_equal(result, expected)