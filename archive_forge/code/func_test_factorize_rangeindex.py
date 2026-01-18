from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('sort', [True, False])
def test_factorize_rangeindex(self, sort):
    ri = pd.RangeIndex.from_range(range(10))
    expected = (np.arange(10, dtype=np.intp), ri)
    result = algos.factorize(ri, sort=sort)
    tm.assert_numpy_array_equal(result[0], expected[0])
    tm.assert_index_equal(result[1], expected[1], exact=True)
    result = ri.factorize(sort=sort)
    tm.assert_numpy_array_equal(result[0], expected[0])
    tm.assert_index_equal(result[1], expected[1], exact=True)