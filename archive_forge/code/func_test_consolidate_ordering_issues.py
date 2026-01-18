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
def test_consolidate_ordering_issues(self, mgr):
    mgr.iset(mgr.items.get_loc('f'), np.random.default_rng(2).standard_normal(N))
    mgr.iset(mgr.items.get_loc('d'), np.random.default_rng(2).standard_normal(N))
    mgr.iset(mgr.items.get_loc('b'), np.random.default_rng(2).standard_normal(N))
    mgr.iset(mgr.items.get_loc('g'), np.random.default_rng(2).standard_normal(N))
    mgr.iset(mgr.items.get_loc('h'), np.random.default_rng(2).standard_normal(N))
    cons = mgr.consolidate()
    assert cons.nblocks == 4
    cons = mgr.consolidate().get_numeric_data()
    assert cons.nblocks == 1
    assert isinstance(cons.blocks[0].mgr_locs, BlockPlacement)
    tm.assert_numpy_array_equal(cons.blocks[0].mgr_locs.as_array, np.arange(len(cons.items), dtype=np.intp))