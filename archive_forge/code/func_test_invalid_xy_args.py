from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('x,y,lbl', [(['B', 'C'], 'A', 'a'), (['A'], ['B', 'C'], ['b', 'c'])])
def test_invalid_xy_args(self, x, y, lbl):
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    with pytest.raises(ValueError, match='x must be a label or position'):
        df.plot(x=x, y=y, label=lbl)