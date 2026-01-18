from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('ddof', range(3))
def test_nanvar(self, ddof, skipna):
    self.check_funs(nanops.nanvar, np.var, skipna, allow_complex=False, allow_date=False, allow_obj='convert', ddof=ddof)