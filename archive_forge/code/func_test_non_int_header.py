from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('header', [['a', 'b'], 'string_header'])
def test_non_int_header(all_parsers, header):
    msg = 'header must be integer or list of integers'
    data = '1,2\n3,4'
    parser = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=header)