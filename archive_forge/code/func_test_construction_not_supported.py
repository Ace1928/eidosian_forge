import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('subtype', [CategoricalDtype(list('abc'), False), CategoricalDtype(list('wxyz'), True), object, str, '<U10', 'interval[category]', 'interval[object]'])
def test_construction_not_supported(self, subtype):
    msg = 'category, object, and string subtypes are not supported for IntervalDtype'
    with pytest.raises(TypeError, match=msg):
        IntervalDtype(subtype)