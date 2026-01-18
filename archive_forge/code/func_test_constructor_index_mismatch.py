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
@pytest.mark.parametrize('input', [[1, 2, 3], (1, 2, 3), list(range(3)), Categorical(['a', 'b', 'a']), (i for i in range(3)), (x for x in range(3))])
def test_constructor_index_mismatch(self, input):
    msg = 'Length of values \\(3\\) does not match length of index \\(4\\)'
    with pytest.raises(ValueError, match=msg):
        Series(input, index=np.arange(4))