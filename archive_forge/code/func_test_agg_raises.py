from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
def test_agg_raises():
    df = DataFrame({'A': [0, 1], 'B': [1, 2]})
    msg = 'Must provide'
    with pytest.raises(TypeError, match=msg):
        df.agg()