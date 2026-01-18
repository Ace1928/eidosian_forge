import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_arg_for_errors(self):
    df = DataFrame([1, 2, 3])
    msg = "Expected value of kwarg 'errors' to be one of ['raise', 'ignore']. Supplied value is 'True'"
    with pytest.raises(ValueError, match=re.escape(msg)):
        df.astype(np.float64, errors=True)
    df.astype(np.int8, errors='ignore')