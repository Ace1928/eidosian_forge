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
def test_iget(self):
    cols = Index(list('abc'))
    values = np.random.default_rng(2).random((3, 3))
    block = new_block(values=values.copy(), placement=BlockPlacement(np.arange(3, dtype=np.intp)), ndim=values.ndim)
    mgr = BlockManager(blocks=(block,), axes=[cols, Index(np.arange(3))])
    tm.assert_almost_equal(mgr.iget(0).internal_values(), values[0])
    tm.assert_almost_equal(mgr.iget(1).internal_values(), values[1])
    tm.assert_almost_equal(mgr.iget(2).internal_values(), values[2])