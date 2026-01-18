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
@pytest.mark.parametrize('int_dtype', [int, np.int16, np.int32, np.int64])
def test_as_json_table_type_int_dtypes(self, int_dtype):
    assert as_json_table_type(int_dtype) == 'integer'