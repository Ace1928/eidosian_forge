from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('thousands', ['_', None])
@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_1000_sep_decimal_float_precision(request, c_parser_only, numeric_decimal, float_precision, thousands):
    decimal_number_check(request, c_parser_only, numeric_decimal, thousands, float_precision)
    text, value = numeric_decimal
    text = ' ' + text + ' '
    if isinstance(value, str):
        value = ' ' + value + ' '
    decimal_number_check(request, c_parser_only, (text, value), thousands, float_precision)