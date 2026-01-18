from io import (
import numpy as np
import pytest
import pandas._libs.parsers as parser
from pandas._libs.parsers import TextReader
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import (
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs
def test_empty_csv_input(self):
    with read_csv(StringIO(), chunksize=20, header=None, names=['a', 'b', 'c']) as df:
        assert isinstance(df, TextFileReader)