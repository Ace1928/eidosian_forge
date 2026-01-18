from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_mixed_datetime64(self):
    df = DataFrame({'A': [1, 2], 'B': ['2012-01-01', '2012-01-02']})
    df['B'] = pd.to_datetime(df.B)
    result = repr(df.loc[0])
    assert '2012-01-01' in result