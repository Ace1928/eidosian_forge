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
def test_dataframe_raises():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    msg = "Cannot pass DataFrame to 'pandas.array'"
    with pytest.raises(TypeError, match=msg):
        pd.array(df)