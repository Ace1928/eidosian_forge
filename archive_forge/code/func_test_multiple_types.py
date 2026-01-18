import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def test_multiple_types(self, file03):
    data_csv = pd.read_csv(file03.replace('.xpt', '.csv'))
    data = read_sas(file03, encoding='utf-8')
    tm.assert_frame_equal(data, data_csv)