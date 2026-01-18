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
def test_nansem(self, ddof, skipna):
    sp_stats = pytest.importorskip('scipy.stats')
    with np.errstate(invalid='ignore'):
        self.check_funs(nanops.nansem, sp_stats.sem, skipna, allow_complex=False, allow_date=False, allow_tdelta=False, allow_obj='convert', ddof=ddof)