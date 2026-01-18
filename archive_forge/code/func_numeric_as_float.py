import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas
def numeric_as_float(data):
    for v in data.columns:
        if data[v].dtype is np.dtype('int64'):
            data[v] = data[v].astype(np.float64)