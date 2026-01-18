import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def create_h5_and_return_checksum(tmp_path, track_times):
    path = tmp_path / setup_path
    df = DataFrame({'a': [1]})
    with HDFStore(path, mode='w') as hdf:
        hdf.put('table', df, format='table', data_columns=True, index=None, track_times=track_times)
    return checksum(path)