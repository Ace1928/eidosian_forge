import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
def test_astype_invalid_dtype(self, index):
    msg = 'data type ["\']fake_dtype["\'] not understood'
    with pytest.raises(TypeError, match=msg):
        index.astype('fake_dtype')