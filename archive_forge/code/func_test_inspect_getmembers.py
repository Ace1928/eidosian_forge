from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_inspect_getmembers(self):
    pytest.importorskip('jinja2')
    df = DataFrame()
    msg = 'DataFrame._data is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
        inspect.getmembers(df)