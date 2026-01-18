from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
def test_empty_fancy_raises(self, index):
    empty_farr = np.array([], dtype=np.float64)
    empty_index = type(index)([], dtype=index.dtype)
    assert index[[]].identical(empty_index)
    msg = 'arrays used as indices must be of integer'
    with pytest.raises(IndexError, match=msg):
        index[empty_farr]