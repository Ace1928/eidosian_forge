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
def test_iterator_read_too_much(self, dirpath):
    fname = os.path.join(dirpath, 'test1.sas7bdat')
    with pd.read_sas(fname, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
        d1 = rdr.read(rdr.row_count + 20)
    with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
        d2 = rdr.read(rdr.row_count + 20)
    tm.assert_frame_equal(d1, d2)