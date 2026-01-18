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
@pytest.mark.parametrize('subplots, expected_msg', [(123, 'subplots should be a bool or an iterable'), ('a', 'each entry should be a list/tuple'), ((1,), 'each entry should be a list/tuple'), (('a',), 'each entry should be a list/tuple')])
def test_group_subplot_bad_input(self, subplots, expected_msg):
    d = {'a': np.arange(10), 'b': np.arange(10)}
    df = DataFrame(d)
    with pytest.raises(ValueError, match=expected_msg):
        df.plot(subplots=subplots)