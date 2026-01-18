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
@pytest.mark.parametrize('index', ['string'], indirect=True)
def test_booleanindex(self, index):
    bool_index = np.ones(len(index), dtype=bool)
    bool_index[5:30:2] = False
    sub_index = index[bool_index]
    for i, val in enumerate(sub_index):
        assert sub_index.get_loc(val) == i
    sub_index = index[list(bool_index)]
    for i, val in enumerate(sub_index):
        assert sub_index.get_loc(val) == i