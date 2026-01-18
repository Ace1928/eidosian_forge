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
def test_0x00_control_byte(datapath):
    fname = datapath('io', 'sas', 'data', '0x00controlbyte.sas7bdat.bz2')
    df = next(pd.read_sas(fname, chunksize=11000))
    assert df.shape == (11000, 20)