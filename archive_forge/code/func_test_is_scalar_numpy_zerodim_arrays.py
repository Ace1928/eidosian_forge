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
@pytest.mark.parametrize('zerodim', [np.array(1), np.array('foobar'), np.array(np.datetime64('2014-01-01')), np.array(np.timedelta64(1, 'h')), np.array(np.datetime64('NaT'))])
def test_is_scalar_numpy_zerodim_arrays(self, zerodim):
    assert not is_scalar(zerodim)
    assert is_scalar(lib.item_from_zerodim(zerodim))