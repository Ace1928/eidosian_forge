import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@staticmethod
def call_op(df, other, flex: bool, opname: str):
    if flex:
        op = lambda x, y: getattr(x, opname)(y)
        op.__name__ = opname
    else:
        op = getattr(operator, opname)
    with option_context('compute.use_numexpr', False):
        expected = op(df, other)
    expr.get_test_result()
    result = op(df, other)
    return (result, expected)