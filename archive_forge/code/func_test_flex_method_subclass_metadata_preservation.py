from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators):

    class MySeries(Series):
        _metadata = ['x']

        @property
        def _constructor(self):
            return MySeries
    opname = all_arithmetic_operators
    op = getattr(Series, opname)
    m = MySeries([1, 2, 3], name='test')
    m.x = 42
    result = op(m, 1)
    assert result.x == 42