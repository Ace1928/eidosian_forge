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
def test_loc_iloc_setitem_mask_single_value_in_categories(self, orig, exp_single_cats_value, indexer):
    df = orig.copy()
    mask = df.index == 'j'
    key = 0
    if indexer is tm.loc:
        key = df.columns[key]
    indexer(df)[mask, key] = 'b'
    tm.assert_frame_equal(df, exp_single_cats_value)