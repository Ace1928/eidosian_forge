import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_from_tzaware_object_array(self):
    dti = date_range('2016-04-05 04:30', periods=3, tz='UTC')
    data = dti._data.astype(object).reshape(1, -1)
    df = DataFrame(data)
    assert df.shape == (1, 3)
    assert (df.dtypes == dti.dtype).all()
    assert (df == dti).all().all()