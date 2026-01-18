from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
def test_loc_iloc_setitem_list_of_lists(self, orig, exp_multi_row, indexer):
    df = orig.copy()
    key = slice(2, 4)
    if indexer is tm.loc:
        key = slice('j', 'k')
    indexer(df)[key, :] = [['b', 2], ['b', 2]]
    tm.assert_frame_equal(df, exp_multi_row)
    df = orig.copy()
    with pytest.raises(TypeError, match=msg1):
        indexer(df)[key, :] = [['c', 2], ['c', 2]]