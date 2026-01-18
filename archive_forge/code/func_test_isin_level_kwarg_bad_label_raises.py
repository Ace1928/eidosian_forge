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
@pytest.mark.parametrize('label', [1.0, 'foobar', 'xyzzy', np.nan])
def test_isin_level_kwarg_bad_label_raises(self, label, index):
    if isinstance(index, MultiIndex):
        index = index.rename(['foo', 'bar'] + index.names[2:])
        msg = f"'Level {label} not found'"
    else:
        index = index.rename('foo')
        msg = f'Requested level \\({label}\\) does not match index name \\(foo\\)'
    with pytest.raises(KeyError, match=msg):
        index.isin([], level=label)