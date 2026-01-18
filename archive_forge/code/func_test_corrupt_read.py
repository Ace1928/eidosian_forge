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
def test_corrupt_read(datapath):
    fname = datapath('io', 'sas', 'data', 'corrupt.sas7bdat')
    msg = "'SAS7BDATReader' object has no attribute 'row_count'"
    with pytest.raises(AttributeError, match=msg):
        pd.read_sas(fname)