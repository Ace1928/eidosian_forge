from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_bool_dep(self) -> None:
    msg_warn = 'DataFrame.bool is now deprecated and will be removed in future version of pandas'
    with tm.assert_produces_warning(FutureWarning, match=msg_warn):
        DataFrame({'col': [False]}).bool()