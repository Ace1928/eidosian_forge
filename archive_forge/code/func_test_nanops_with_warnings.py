from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('nan_op,np_op', [(nanops.nanmin, np.min), (nanops.nanmax, np.max)])
def test_nanops_with_warnings(self, nan_op, np_op, skipna):
    self.check_funs(nan_op, np_op, skipna, allow_obj=False)