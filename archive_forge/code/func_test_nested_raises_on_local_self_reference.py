import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_nested_raises_on_local_self_reference(self, engine, parser):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
        df.query('df > 0', engine=engine, parser=parser)