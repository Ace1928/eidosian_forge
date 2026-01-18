import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def test2_binary(self, file02):
    data_csv = pd.read_csv(file02.replace('.xpt', '.csv'))
    numeric_as_float(data_csv)
    with open(file02, 'rb') as fd:
        data = read_sas(fd, format='xport')
    tm.assert_frame_equal(data, data_csv)