from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('header', [True, False])
def test_bool_header_arg(all_parsers, header):
    parser = all_parsers
    data = 'MyColumn\na\nb\na\nb'
    msg = 'Passing a bool to header is invalid'
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), header=header)