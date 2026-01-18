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
def test_data_frame_size_after_to_json(self):
    df = DataFrame({'a': [str(1)]})
    size_before = df.memory_usage(index=True, deep=True).sum()
    df.to_json()
    size_after = df.memory_usage(index=True, deep=True).sum()
    assert size_before == size_after