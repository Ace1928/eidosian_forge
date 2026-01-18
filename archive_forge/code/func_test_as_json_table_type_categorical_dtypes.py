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
def test_as_json_table_type_categorical_dtypes(self):
    assert as_json_table_type(pd.Categorical(['a']).dtype) == 'any'
    assert as_json_table_type(CategoricalDtype()) == 'any'