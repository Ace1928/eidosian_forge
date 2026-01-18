from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
def test_empty_frame_roundtrip(self):
    df = DataFrame(columns=['a', 'b', 'c'])
    expected = df.copy()
    out = StringIO(df.to_json(orient='table'))
    result = pd.read_json(out, orient='table')
    tm.assert_frame_equal(expected, result)