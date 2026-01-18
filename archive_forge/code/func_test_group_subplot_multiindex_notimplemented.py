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
def test_group_subplot_multiindex_notimplemented(self):
    df = DataFrame(np.eye(2), columns=MultiIndex.from_tuples([(0, 1), (1, 2)]))
    msg = 'An iterable subplots for a DataFrame with a MultiIndex'
    with pytest.raises(NotImplementedError, match=msg):
        df.plot(subplots=[(0, 1)])