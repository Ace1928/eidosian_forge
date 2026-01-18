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
@pytest.mark.parametrize('vals,dtype', [([1, 2, 3, 4, 5], 'int'), ([1.1, np.nan, 2.2, 3.0], 'float'), (['A', 'B', 'C', np.nan], 'obj')])
def test_constructor_simple_new(self, vals, dtype):
    index = Index(vals, name=dtype)
    result = index._simple_new(index.values, dtype)
    tm.assert_index_equal(result, index)