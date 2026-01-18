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
def test_decimal_and_exponential(request, python_parser_only, numeric_decimal, thousands):
    decimal_number_check(request, python_parser_only, numeric_decimal, thousands, None)