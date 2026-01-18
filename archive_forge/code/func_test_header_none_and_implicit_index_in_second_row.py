from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_header_none_and_implicit_index_in_second_row(all_parsers):
    parser = all_parsers
    data = 'x,1\ny,2,5\nz,3\n'
    with pytest.raises(ParserError, match='Expected 2 fields in line 2, saw 3'):
        parser.read_csv(StringIO(data), names=['a', 'b'], header=None)