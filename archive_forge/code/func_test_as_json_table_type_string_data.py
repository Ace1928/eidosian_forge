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
@pytest.mark.parametrize('str_data', [pd.Series(['a', 'b'], dtype=object), pd.Index(['a', 'b'], dtype=object)])
def test_as_json_table_type_string_data(self, str_data):
    assert as_json_table_type(str_data.dtype) == 'string'