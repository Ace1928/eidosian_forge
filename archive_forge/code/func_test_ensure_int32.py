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
def test_ensure_int32():
    values = np.arange(10, dtype=np.int32)
    result = ensure_int32(values)
    assert result.dtype == np.int32
    values = np.arange(10, dtype=np.int64)
    result = ensure_int32(values)
    assert result.dtype == np.int32