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
@pytest.mark.parametrize('unit', ['ps', 'as', 'fs', 'Y', 'M', 'W', 'D', 'h', 'm'])
@pytest.mark.parametrize('kind', ['m', 'M'])
def test_constructor_generic_timestamp_bad_frequency(self, kind, unit):
    dtype = f'{kind}8[{unit}]'
    msg = 'dtype=.* is not supported. Supported resolutions are'
    with pytest.raises(TypeError, match=msg):
        Series([], dtype=dtype)
    with pytest.raises(TypeError, match=msg):
        DataFrame([[0]], dtype=dtype)