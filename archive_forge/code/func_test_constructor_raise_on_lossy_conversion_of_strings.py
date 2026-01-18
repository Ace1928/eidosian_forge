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
def test_constructor_raise_on_lossy_conversion_of_strings(self):
    if not np_version_gt2:
        raises = pytest.raises(ValueError, match='string values cannot be losslessly cast to int8')
    else:
        raises = pytest.raises(OverflowError, match='The elements provided in the data')
    with raises:
        Series(['128'], dtype='int8')