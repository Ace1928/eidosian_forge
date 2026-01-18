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
def test_no_nonsense_name(float_frame):
    s = float_frame['C'].copy()
    s.name = None
    result = s.groupby(float_frame['A']).agg('sum')
    assert result.name is None