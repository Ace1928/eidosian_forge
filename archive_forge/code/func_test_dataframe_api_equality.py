import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
def test_dataframe_api_equality():
    modin_dir = [obj for obj in dir(pd.DataFrame) if obj[0] != '_']
    pandas_dir = [obj for obj in dir(pandas.DataFrame) if obj[0] != '_']
    ignore_in_pandas = ['timetuple']
    ignore_in_modin = ['modin']
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin - set(ignore_in_pandas)), 'Differences found in API: {}'.format(len(missing_from_modin - set(ignore_in_pandas)))
    assert not len(set(modin_dir) - set(ignore_in_modin) - set(pandas_dir)), 'Differences found in API: {}'.format(set(modin_dir) - set(pandas_dir))
    allowed_different = ['modin']
    assert_parameters_eq((pandas.DataFrame, pd.DataFrame), modin_dir, allowed_different)