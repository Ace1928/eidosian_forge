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
@pytest.mark.parametrize('td_dtype', [np.dtype('<m8[ns]')])
def test_as_json_table_type_timedelta_dtypes(self, td_dtype):
    assert as_json_table_type(td_dtype) == 'duration'