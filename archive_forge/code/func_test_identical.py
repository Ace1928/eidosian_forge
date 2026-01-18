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
def test_identical(self):
    i1 = Index(['a', 'b', 'c'])
    i2 = Index(['a', 'b', 'c'])
    assert i1.identical(i2)
    i1 = i1.rename('foo')
    assert i1.equals(i2)
    assert not i1.identical(i2)
    i2 = i2.rename('foo')
    assert i1.identical(i2)
    i3 = Index([('a', 'a'), ('a', 'b'), ('b', 'a')])
    i4 = Index([('a', 'a'), ('a', 'b'), ('b', 'a')], tupleize_cols=False)
    assert not i3.identical(i4)