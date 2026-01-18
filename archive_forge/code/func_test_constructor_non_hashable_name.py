from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_non_hashable_name(self, index_flat):
    index = index_flat
    message = 'Index.name must be a hashable type'
    renamed = [['1']]
    with pytest.raises(TypeError, match=message):
        index.rename(name=renamed)
    with pytest.raises(TypeError, match=message):
        index.set_names(names=renamed)