from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_categories_raises(self):
    ser = Series(['a', 'b', 'a'])
    with pytest.raises(TypeError, match='got an unexpected'):
        ser.astype('category', categories=['a', 'b'], ordered=True)