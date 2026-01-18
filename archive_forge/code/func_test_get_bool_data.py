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
def test_get_bool_data(self, using_copy_on_write):
    mgr = create_mgr('int: int; float: float; complex: complex;str: object; bool: bool; obj: object; dt: datetime', item_shape=(3,))
    mgr.iset(6, np.array([True, False, True], dtype=np.object_))
    bools = mgr.get_bool_data()
    tm.assert_index_equal(bools.items, Index(['bool']))
    tm.assert_almost_equal(mgr.iget(mgr.items.get_loc('bool')).internal_values(), bools.iget(bools.items.get_loc('bool')).internal_values())
    bools.iset(0, np.array([True, False, True]), inplace=True)
    if using_copy_on_write:
        tm.assert_numpy_array_equal(mgr.iget(mgr.items.get_loc('bool')).internal_values(), np.array([True, True, True]))
    else:
        tm.assert_numpy_array_equal(mgr.iget(mgr.items.get_loc('bool')).internal_values(), np.array([True, False, True]))