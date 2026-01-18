from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('df', [DataFrame({'A': ['a', None], 'B': ['c', 'd']})])
@pytest.mark.parametrize('method', ['min', 'max', 'sum'])
def test_mixed_column_raises(df, method, using_infer_string):
    if method == 'sum':
        msg = 'can only concatenate str \\(not "int"\\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        getattr(df, method)()