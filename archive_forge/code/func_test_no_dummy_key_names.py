from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_no_dummy_key_names(df):
    result = df.groupby(df['A'].values).sum()
    assert result.index.name is None
    result = df.groupby([df['A'].values, df['B'].values]).sum()
    assert result.index.names == (None, None)