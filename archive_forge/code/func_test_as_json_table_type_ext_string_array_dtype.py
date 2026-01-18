from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
from pandas.tests.extension.decimal.array import (
from pandas.io.json._table_schema import (
@pytest.mark.parametrize('string_data', [array(['pandas'], dtype='string'), Series(array(['pandas'], dtype='string'))])
def test_as_json_table_type_ext_string_array_dtype(self, string_data):
    assert as_json_table_type(string_data.dtype) == 'any'