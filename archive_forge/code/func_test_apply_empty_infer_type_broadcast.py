from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_empty_infer_type_broadcast():
    no_cols = DataFrame(index=['a', 'b', 'c'])
    result = no_cols.apply(lambda x: x.mean(), result_type='broadcast')
    assert isinstance(result, DataFrame)