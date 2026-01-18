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
def test_constructor_categorical_with_coercion2(self):
    x = DataFrame([[1, 'John P. Doe'], [2, 'Jane Dove'], [1, 'John P. Doe']], columns=['person_id', 'person_name'])
    x['person_name'] = Categorical(x.person_name)
    expected = x.iloc[0].person_name
    result = x.person_name.iloc[0]
    assert result == expected
    result = x.person_name[0]
    assert result == expected
    result = x.person_name.loc[0]
    assert result == expected