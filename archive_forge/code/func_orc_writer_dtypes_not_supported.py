import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
import numpy as np
import pytest
import pandas as pd
from pandas import read_orc
import pandas._testing as tm
from pandas.core.arrays import StringArray
import pyarrow as pa
@pytest.fixture(params=[np.array([1, 20], dtype='uint64'), pd.Series(['a', 'b', 'a'], dtype='category'), [pd.Interval(left=0, right=2), pd.Interval(left=0, right=5)], [pd.Period('2022-01-03', freq='D'), pd.Period('2022-01-04', freq='D')]])
def orc_writer_dtypes_not_supported(request):
    return pd.DataFrame({'unimpl': request.param})