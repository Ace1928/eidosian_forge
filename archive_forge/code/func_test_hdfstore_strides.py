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
def test_hdfstore_strides(setup_path):
    df = DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
    with ensure_clean_store(setup_path) as store:
        store.put('df', df)
        assert df['a'].values.strides == store['df']['a'].values.strides