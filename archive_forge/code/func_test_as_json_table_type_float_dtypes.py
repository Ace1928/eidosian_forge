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
@pytest.mark.parametrize('float_dtype', [float, np.float16, np.float32, np.float64])
def test_as_json_table_type_float_dtypes(self, float_dtype):
    assert as_json_table_type(float_dtype) == 'number'