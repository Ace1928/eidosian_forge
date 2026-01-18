import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_is_scalar_pandas_containers(self):
    assert not is_scalar(Series(dtype=object))
    assert not is_scalar(Series([1]))
    assert not is_scalar(DataFrame())
    assert not is_scalar(DataFrame([[1]]))
    assert not is_scalar(Index([]))
    assert not is_scalar(Index([1]))
    assert not is_scalar(Categorical([]))
    assert not is_scalar(DatetimeIndex([])._data)
    assert not is_scalar(TimedeltaIndex([])._data)
    assert not is_scalar(DatetimeIndex([])._data.to_period('D'))
    assert not is_scalar(pd.array([1, 2, 3]))