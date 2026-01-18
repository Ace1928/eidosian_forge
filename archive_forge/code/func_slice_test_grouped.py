import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture()
def slice_test_grouped(slice_test_df):
    return slice_test_df.groupby('Group', as_index=False)