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
def test_meta2_page(datapath):
    fname = datapath('io', 'sas', 'data', 'test_meta2_page.sas7bdat')
    df = pd.read_sas(fname)
    assert len(df) == 1000