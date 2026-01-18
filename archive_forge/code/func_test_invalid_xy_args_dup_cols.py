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
@pytest.mark.parametrize('x,y', [('A', 'B'), (['A'], 'B')])
def test_invalid_xy_args_dup_cols(self, x, y):
    df = DataFrame([[1, 3, 5], [2, 4, 6]], columns=list('AAB'))
    with pytest.raises(ValueError, match='x must be a label or position'):
        df.plot(x=x, y=y)