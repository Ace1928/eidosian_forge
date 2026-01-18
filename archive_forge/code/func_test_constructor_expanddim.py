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
def test_constructor_expanddim(self):
    df = DataFrame()
    msg = "'DataFrame' object has no attribute '_constructor_expanddim'"
    with pytest.raises(AttributeError, match=msg):
        df._constructor_expanddim(np.arange(27).reshape(3, 3, 3))