from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def check_missing(res):
    if dtype == 'datetime64[ns]':
        return res is NaT
    elif dtype in ['Int64', 'boolean']:
        return res is pd.NA
    else:
        return isna(res)