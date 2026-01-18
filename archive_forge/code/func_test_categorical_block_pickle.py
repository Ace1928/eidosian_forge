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
def test_categorical_block_pickle(self):
    mgr = create_mgr('a: category')
    mgr2 = tm.round_trip_pickle(mgr)
    tm.assert_frame_equal(DataFrame._from_mgr(mgr, axes=mgr.axes), DataFrame._from_mgr(mgr2, axes=mgr2.axes))
    smgr = create_single_mgr('category')
    smgr2 = tm.round_trip_pickle(smgr)
    tm.assert_series_equal(Series()._constructor_from_mgr(smgr, axes=smgr.axes), Series()._constructor_from_mgr(smgr2, axes=smgr2.axes))