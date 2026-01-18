import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('orient', ['records', 'values'])
def test_index_true_error_to_json(self, orient):
    df = DataFrame([[1, 2], [4, 5]], columns=['a', 'b'])
    msg = "'index=True' is only valid when 'orient' is 'split', 'table', 'index', or 'columns'"
    with pytest.raises(ValueError, match=msg):
        df.to_json(orient=orient, index=True)