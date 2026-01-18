from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('skipfooter', ['foo', 1.5, True])
def test_invalid_skipfooter_non_int(python_parser_only, skipfooter):
    data = 'a\n1\n2'
    parser = python_parser_only
    msg = 'skipfooter must be an integer'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=skipfooter)