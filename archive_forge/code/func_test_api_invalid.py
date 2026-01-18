import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_api_invalid(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    msg = 'Can only append to Tables'
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key='df', append=True, format='f')
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key='df', append=True, format='fixed')
    msg = 'invalid HDFStore format specified \\[foo\\]'
    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key='df', append=True, format='foo')
    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key='df', append=False, format='foo')
    path = ''
    msg = f'File {path} does not exist'
    with pytest.raises(FileNotFoundError, match=msg):
        read_hdf(path, 'df')