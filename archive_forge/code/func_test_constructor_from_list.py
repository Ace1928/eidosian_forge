import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
def test_constructor_from_list():
    pytest.importorskip('pyarrow')
    result = pd.Series(['E'], dtype=StringDtype(storage='pyarrow'))
    assert isinstance(result.dtype, StringDtype)
    assert result.dtype.storage == 'pyarrow'