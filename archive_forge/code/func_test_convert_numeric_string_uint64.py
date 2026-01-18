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
def test_convert_numeric_string_uint64(self, convert_to_masked_nullable):
    result = lib.maybe_convert_numeric(np.array(['uint64'], dtype=object), set(), coerce_numeric=True, convert_to_masked_nullable=convert_to_masked_nullable)
    if convert_to_masked_nullable:
        result = FloatingArray(*result)
    else:
        result = result[0]
    assert np.isnan(result)