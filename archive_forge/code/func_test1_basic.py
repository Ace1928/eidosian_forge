import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
@pytest.mark.slow
def test1_basic(self, file01):
    data_csv = pd.read_csv(file01.replace('.xpt', '.csv'))
    numeric_as_float(data_csv)
    data = read_sas(file01, format='xport')
    tm.assert_frame_equal(data, data_csv)
    num_rows = data.shape[0]
    with read_sas(file01, format='xport', iterator=True) as reader:
        data = reader.read(num_rows + 100)
    assert data.shape[0] == num_rows
    with read_sas(file01, format='xport', iterator=True) as reader:
        data = reader.read(10)
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :])
    with read_sas(file01, format='xport', chunksize=10) as reader:
        data = reader.get_chunk()
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :])
    m = 0
    with read_sas(file01, format='xport', chunksize=100) as reader:
        for x in reader:
            m += x.shape[0]
    assert m == num_rows
    data = read_sas(file01)
    tm.assert_frame_equal(data, data_csv)