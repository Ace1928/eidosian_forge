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
def test_nanmedian(self, skipna):
    self.check_funs(nanops.nanmedian, np.median, skipna, allow_complex=False, allow_date=False, allow_obj='convert')