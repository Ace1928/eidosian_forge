import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('func', [lambda s: np.percentile(s, q=0.9), myfunc])
def test_lambda_named_agg(func):
    animals = DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'], 'height': [9.1, 6.0, 9.5, 34.0], 'weight': [7.9, 7.5, 9.9, 198.0]})
    result = animals.groupby('kind').agg(mean_height=('height', 'mean'), perc90=('height', func))
    expected = DataFrame([[9.3, 9.1036], [20.0, 6.252]], columns=['mean_height', 'perc90'], index=Index(['cat', 'dog'], name='kind'))
    tm.assert_frame_equal(result, expected)