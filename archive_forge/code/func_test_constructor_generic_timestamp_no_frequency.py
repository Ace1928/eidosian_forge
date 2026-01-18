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
@pytest.mark.parametrize('dtype', [np.datetime64, np.timedelta64])
def test_constructor_generic_timestamp_no_frequency(self, dtype, request):
    msg = 'dtype has no unit. Please pass in'
    if np.dtype(dtype).name not in ['timedelta64', 'datetime64']:
        mark = pytest.mark.xfail(reason='GH#33890 Is assigned ns unit')
        request.applymarker(mark)
    with pytest.raises(ValueError, match=msg):
        Series([], dtype=dtype)