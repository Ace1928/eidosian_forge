from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_null_in_object_column(nulls_fixture):
    df = DataFrame([[1, nulls_fixture], [2, 'a']], dtype=object)
    result = df.drop_duplicates()
    tm.assert_frame_equal(result, df)