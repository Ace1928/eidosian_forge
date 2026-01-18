import datetime
import decimal
import re
import numpy as np
import pytest
import pytz
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
from pandas.core.arrays import (
from pandas.tests.extension.decimal import (
@pytest.mark.parametrize('data', [np.array(0)])
def test_nd_raises(data):
    with pytest.raises(ValueError, match='NumpyExtensionArray must be 1-dimensional'):
        pd.array(data, dtype='int64')