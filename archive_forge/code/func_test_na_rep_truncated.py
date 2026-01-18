import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_na_rep_truncated(self):
    result = pd.Series(range(8, 12)).to_csv(na_rep='-')
    expected = tm.convert_rows_list_to_csv_str([',0', '0,8', '1,9', '2,10', '3,11'])
    assert result == expected
    result = pd.Series([True, False]).to_csv(na_rep='nan')
    expected = tm.convert_rows_list_to_csv_str([',0', '0,True', '1,False'])
    assert result == expected
    result = pd.Series([1.1, 2.2]).to_csv(na_rep='.')
    expected = tm.convert_rows_list_to_csv_str([',0', '0,1.1', '1,2.2'])
    assert result == expected