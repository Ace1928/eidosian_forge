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
@pytest.mark.parametrize('indexer', [tm.loc, tm.iloc, tm.at, tm.iat])
def test_loc_iloc_at_iat_setitem_single_value_in_categories(self, orig, exp_single_cats_value, indexer):
    df = orig.copy()
    key = (2, 0)
    if indexer in [tm.loc, tm.at]:
        key = (df.index[2], df.columns[0])
    indexer(df)[key] = 'b'
    tm.assert_frame_equal(df, exp_single_cats_value)
    with pytest.raises(TypeError, match=msg1):
        indexer(df)[key] = 'c'