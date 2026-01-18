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
@pytest.mark.parametrize('bool_type', [bool, np.bool_])
def test_as_json_table_type_bool_data(self, bool_type):
    bool_data = [True, False]
    assert as_json_table_type(np.array(bool_data, dtype=bool_type).dtype) == 'boolean'