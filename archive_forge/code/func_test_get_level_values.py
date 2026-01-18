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
@pytest.mark.parametrize('name,level', [(None, 0), ('a', 'a')])
def test_get_level_values(self, index, name, level):
    expected = index.copy()
    if name:
        expected.name = name
    result = expected.get_level_values(level)
    tm.assert_index_equal(result, expected)