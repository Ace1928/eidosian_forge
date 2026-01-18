from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_inference_consistency():
    v = date.today()
    arr = [v, v]
    idx = Index(arr)
    assert idx.dtype == object
    mi = MultiIndex.from_arrays([arr])
    lev = mi.levels[0]
    assert lev.dtype == object
    mi = MultiIndex.from_product([arr])
    lev = mi.levels[0]
    assert lev.dtype == object
    mi = MultiIndex.from_tuples([(x,) for x in arr])
    lev = mi.levels[0]
    assert lev.dtype == object