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
def test_constructor_signed_int_overflow_raises(self):
    if np_version_gt2:
        msg = 'The elements provided in the data cannot all be casted to the dtype'
        err = OverflowError
    else:
        msg = 'Values are too large to be losslessly converted'
        err = ValueError
    with pytest.raises(err, match=msg):
        Series([1, 200, 923442], dtype='int8')
    with pytest.raises(err, match=msg):
        Series([1, 200, 923442], dtype='uint8')