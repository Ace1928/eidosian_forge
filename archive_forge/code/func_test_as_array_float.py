from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.mark.parametrize('mgr_string, dtype', [('c: f4; d: f2', np.float32), ('c: f4; d: f2; e: f8', np.float64)])
def test_as_array_float(self, mgr_string, dtype):
    mgr = create_mgr(mgr_string)
    assert mgr.as_array().dtype == dtype