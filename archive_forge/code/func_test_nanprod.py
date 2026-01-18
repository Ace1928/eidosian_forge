from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nanprod(self, skipna):
    self.check_funs(nanops.nanprod, np.prod, skipna, allow_date=False, allow_tdelta=False, empty_targfunc=np.nanprod)