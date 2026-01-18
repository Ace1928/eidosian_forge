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
def test_copy_name(self, simple_index):
    index = simple_index
    first = type(index)(index, copy=True, name='mario')
    second = type(first)(first, copy=False)
    assert first is not second
    tm.assert_index_equal(first, second)
    assert first.name == 'mario'
    assert second.name == 'mario'
    s1 = Series(2, index=first)
    s2 = Series(3, index=second[:-1])
    s3 = s1 * s2
    assert s3.index.name == 'mario'