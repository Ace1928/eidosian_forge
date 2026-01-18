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
@pytest.fixture
def datetime_frame(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=30, freq='B'))
    df.index = df.index._with_freq(None)
    return df