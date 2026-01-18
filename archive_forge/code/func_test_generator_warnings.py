import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_generator_warnings(self):
    sp_arr = SparseArray([1, 2, 3])
    with tm.assert_produces_warning(None):
        for _ in sp_arr:
            pass