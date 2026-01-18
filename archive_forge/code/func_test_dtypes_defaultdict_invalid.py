from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtypes_defaultdict_invalid(all_parsers):
    data = 'a,b\n1,2\n'
    dtype = defaultdict(lambda: 'invalid_dtype', a='int64')
    parser = all_parsers
    with pytest.raises(TypeError, match='not understood'):
        parser.read_csv(StringIO(data), dtype=dtype)