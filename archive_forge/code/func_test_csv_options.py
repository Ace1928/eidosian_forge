import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_csv_options(fsspectest):
    df = DataFrame({'a': [0]})
    df.to_csv('testmem://test/test.csv', storage_options={'test': 'csv_write'}, index=False)
    assert fsspectest.test[0] == 'csv_write'
    read_csv('testmem://test/test.csv', storage_options={'test': 'csv_read'})
    assert fsspectest.test[0] == 'csv_read'