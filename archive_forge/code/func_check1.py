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
def check1(exp, kwarg):
    result = merge(left, right, how='inner', **kwarg)
    tm.assert_frame_equal(result, exp)
    result = merge(left, right, how='right', **kwarg)
    tm.assert_frame_equal(result, exp)