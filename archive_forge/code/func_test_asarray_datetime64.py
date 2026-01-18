import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_asarray_datetime64(self):
    s = SparseArray(pd.to_datetime(['2012', None, None, '2013']))
    np.asarray(s)