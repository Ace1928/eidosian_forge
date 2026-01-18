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
def test_attrs_deepcopy(self):
    df = DataFrame({'A': [2, 3]})
    assert df.attrs == {}
    df.attrs['tags'] = {'spam', 'ham'}
    result = df.rename(columns=str)
    assert result.attrs == df.attrs
    assert result.attrs['tags'] is not df.attrs['tags']