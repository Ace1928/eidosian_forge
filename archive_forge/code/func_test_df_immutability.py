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
def test_df_immutability():
    """
    Verify that modifications of the source data doesn't propagate to Modin's DataFrame objects.
    """
    src_data = pandas.DataFrame({'a': [1]})
    md_df = pd.DataFrame(src_data)
    src_data.iloc[0, 0] = 100
    assert md_df._to_pandas().iloc[0, 0] == 1