import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dataframe_consortium() -> None:
    """
    Test some basic methods of the dataframe consortium standard.

    Full testing is done at https://github.com/data-apis/dataframe-api-compat,
    this is just to check that the entry point works as expected.
    """
    pytest.importorskip('dataframe_api_compat')
    df_pd = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df = df_pd.__dataframe_consortium_standard__()
    result_1 = df.get_column_names()
    expected_1 = ['a', 'b']
    assert result_1 == expected_1
    ser = Series([1, 2, 3], name='a')
    col = ser.__column_consortium_standard__()
    assert col.name == 'a'