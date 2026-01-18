import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_int_raises_exception(frame_random_data_integer_multi_index):
    df = frame_random_data_integer_multi_index
    with pytest.raises(KeyError, match='^3$'):
        df.loc[3]