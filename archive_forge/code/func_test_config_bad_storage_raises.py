import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
def test_config_bad_storage_raises():
    msg = re.escape('Value must be one of python|pyarrow')
    with pytest.raises(ValueError, match=msg):
        pd.options.mode.string_storage = 'foo'