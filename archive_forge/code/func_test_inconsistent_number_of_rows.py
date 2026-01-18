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
def test_inconsistent_number_of_rows(datapath):
    fname = datapath('io', 'sas', 'data', 'load_log.sas7bdat')
    df = pd.read_sas(fname, encoding='latin-1')
    assert len(df) == 2097