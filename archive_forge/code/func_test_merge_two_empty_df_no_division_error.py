from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_two_empty_df_no_division_error(self):
    a = DataFrame({'a': [], 'b': [], 'c': []})
    with np.errstate(divide='raise'):
        merge(a, a, on=('a', 'b'))