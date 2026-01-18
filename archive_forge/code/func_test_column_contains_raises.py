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
def test_column_contains_raises(self, float_frame):
    with pytest.raises(TypeError, match="unhashable type: 'Index'"):
        float_frame.columns in float_frame