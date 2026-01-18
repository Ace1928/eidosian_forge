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
@pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
@pytest.mark.parametrize('coerce_numeric', [True, False])
@pytest.mark.parametrize('infinity', ['inf', 'inF', 'iNf', 'Inf', 'iNF', 'InF', 'INf', 'INF'])
@pytest.mark.parametrize('prefix', ['', '-', '+'])
def test_maybe_convert_numeric_infinities(self, coerce_numeric, infinity, prefix, convert_to_masked_nullable):
    result, _ = lib.maybe_convert_numeric(np.array([prefix + infinity], dtype=object), na_values={'', 'NULL', 'nan'}, coerce_numeric=coerce_numeric, convert_to_masked_nullable=convert_to_masked_nullable)
    expected = np.array([np.inf if prefix in ['', '+'] else -np.inf])
    tm.assert_numpy_array_equal(result, expected)