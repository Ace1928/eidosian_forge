from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_invalid_dtype_per_column(all_parsers):
    parser = all_parsers
    data = 'one,two\n1,2.5\n2,3.5\n3,4.5\n4,5.5'
    with pytest.raises(TypeError, match='data type ["\']foo["\'] not understood'):
        parser.read_csv(StringIO(data), dtype={'one': 'foo', 1: 'int'})