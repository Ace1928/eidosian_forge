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
def test_delete_datetimelike(self):
    arr = np.arange(20, dtype='i8').reshape(5, 4).view('m8[ns]')
    df = DataFrame(arr)
    blk = df._mgr.blocks[0]
    assert isinstance(blk.values, TimedeltaArray)
    nb = blk.delete(1)
    assert len(nb) == 2
    assert isinstance(nb[0].values, TimedeltaArray)
    assert isinstance(nb[1].values, TimedeltaArray)
    df = DataFrame(arr.view('M8[ns]'))
    blk = df._mgr.blocks[0]
    assert isinstance(blk.values, DatetimeArray)
    nb = blk.delete([1, 3])
    assert len(nb) == 2
    assert isinstance(nb[0].values, DatetimeArray)
    assert isinstance(nb[1].values, DatetimeArray)