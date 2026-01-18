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
def test_loc_named_tuple_for_midx(self):
    df = DataFrame(index=MultiIndex.from_product([['A', 'B'], ['a', 'b', 'c']], names=['first', 'second']))
    indexer_tuple = namedtuple('Indexer', df.index.names)
    idxr = indexer_tuple(first='A', second=['a', 'b'])
    result = df.loc[idxr, :]
    expected = DataFrame(index=MultiIndex.from_tuples([('A', 'a'), ('A', 'b')], names=['first', 'second']))
    tm.assert_frame_equal(result, expected)