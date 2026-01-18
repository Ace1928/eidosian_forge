from __future__ import annotations
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.astype import astype_array
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
import pandas as pd
import pandas._testing as tm
from pandas.api.types import pandas_dtype
from pandas.arrays import SparseArray
def get_is_dtype_funcs():
    """
    Get all functions in pandas.core.dtypes.common that
    begin with 'is_' and end with 'dtype'

    """
    fnames = [f for f in dir(com) if f.startswith('is_') and f.endswith('dtype')]
    fnames.remove('is_string_or_object_np_dtype')
    return [getattr(com, fname) for fname in fnames]