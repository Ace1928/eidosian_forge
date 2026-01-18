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
def test_default_handler_raises(self):
    msg = 'raisin'

    def my_handler_raises(obj):
        raise TypeError(msg)
    with pytest.raises(TypeError, match=msg):
        DataFrame({'a': [1, 2, object()]}).to_json(default_handler=my_handler_raises)
    with pytest.raises(TypeError, match=msg):
        DataFrame({'a': [1, 2, complex(4, -5)]}).to_json(default_handler=my_handler_raises)