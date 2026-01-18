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
def test_is_(self):
    ind = Index(range(10))
    assert ind.is_(ind)
    assert ind.is_(ind.view().view().view().view())
    assert not ind.is_(Index(range(10)))
    assert not ind.is_(ind.copy())
    assert not ind.is_(ind.copy(deep=False))
    assert not ind.is_(ind[:])
    assert not ind.is_(np.array(range(10)))
    assert ind.is_(ind.view())
    ind2 = ind.view()
    ind2.name = 'bob'
    assert ind.is_(ind2)
    assert ind2.is_(ind)
    assert not ind.is_(Index(ind.values))
    arr = np.array(range(1, 11))
    ind1 = Index(arr, copy=False)
    ind2 = Index(arr, copy=False)
    assert not ind1.is_(ind2)