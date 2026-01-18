from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_product_empty_zero_levels():
    msg = 'Must pass non-zero number of levels/codes'
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_product([])