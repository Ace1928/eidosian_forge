from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
def test_generic_errors(self, constructor):
    """
        override the base class implementation since errors are handled
        differently; checks unnecessary since caught at the Interval level
        """