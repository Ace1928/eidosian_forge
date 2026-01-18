import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_level_setting_resets_attributes():
    ind = MultiIndex.from_arrays([['A', 'A', 'B', 'B', 'B'], [1, 2, 1, 2, 3]])
    assert ind.is_monotonic_increasing
    ind = ind.set_levels([['A', 'B'], [1, 3, 2]])
    assert not ind.is_monotonic_increasing