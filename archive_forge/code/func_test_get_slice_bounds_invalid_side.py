import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_slice_bounds_invalid_side(self):
    with pytest.raises(ValueError, match='Invalid value for side kwarg'):
        Index([]).get_slice_bound('a', side='middle')