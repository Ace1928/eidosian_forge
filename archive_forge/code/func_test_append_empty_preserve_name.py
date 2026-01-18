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
@pytest.mark.parametrize('name,expected', [('foo', 'foo'), ('bar', None)])
def test_append_empty_preserve_name(self, name, expected):
    left = Index([], name='foo')
    right = Index([1, 2, 3], name=name)
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = left.append(right)
    assert result.name == expected