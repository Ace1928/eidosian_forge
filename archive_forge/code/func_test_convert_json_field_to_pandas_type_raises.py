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
@pytest.mark.parametrize('inp', ['geopoint', 'geojson', 'fake_type'])
def test_convert_json_field_to_pandas_type_raises(self, inp):
    field = {'type': inp}
    with pytest.raises(ValueError, match=f'Unsupported or invalid field type: {inp}'):
        convert_json_field_to_pandas_type(field)