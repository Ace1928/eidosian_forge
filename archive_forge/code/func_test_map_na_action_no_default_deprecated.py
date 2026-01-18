import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_na_action_no_default_deprecated():
    cat = Categorical(['a', 'b', 'c'])
    msg = "The default value of 'ignore' for the `na_action` parameter in pandas.Categorical.map is deprecated and will be changed to 'None' in a future version. Please set na_action to the desired value to avoid seeing this warning"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        cat.map(lambda x: x)