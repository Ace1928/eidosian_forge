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
def assert_take_ok(mgr, axis, indexer):
    mat = _as_array(mgr)
    taken = mgr.take(indexer, axis)
    tm.assert_numpy_array_equal(np.take(mat, indexer, axis), _as_array(taken), check_dtype=False)
    tm.assert_index_equal(mgr.axes[axis].take(indexer), taken.axes[axis])