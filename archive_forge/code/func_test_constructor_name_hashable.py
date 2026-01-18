from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_name_hashable(self):
    for n in [777, 777.0, 'name', datetime(2001, 11, 11), (1,), 'א']:
        for data in [[1, 2, 3], np.ones(3), {'a': 0, 'b': 1}]:
            s = Series(data, name=n)
            assert s.name == n