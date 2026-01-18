import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_nbytes_integer(self):
    arr = SparseArray([1, 0, 0, 0, 2], kind='integer')
    result = arr.nbytes
    assert result == 24