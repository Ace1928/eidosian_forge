import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
def test_cut_fallback():
    pandas_result = pandas.cut(range(5), 4)
    with warns_that_defaulting_to_pandas():
        modin_result = pd.cut(range(5), 4)
    df_equals(modin_result, pandas_result)