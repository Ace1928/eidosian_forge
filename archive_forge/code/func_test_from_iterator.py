import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
@pytest.mark.slow
def test_from_iterator(self, dirpath, data_test_ix):
    expected, test_ix = data_test_ix
    for k in test_ix:
        fname = os.path.join(dirpath, f'test{k}.sas7bdat')
        with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
            df = rdr.read(2)
            tm.assert_frame_equal(df, expected.iloc[0:2, :])
            df = rdr.read(3)
            tm.assert_frame_equal(df, expected.iloc[2:5, :])