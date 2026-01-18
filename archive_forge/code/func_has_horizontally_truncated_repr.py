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
def has_horizontally_truncated_repr(df):
    try:
        fst_line = np.array(repr(df).splitlines()[0].split())
        cand_col = np.where(fst_line == '...')[0][0]
    except IndexError:
        return False
    r = repr(df)
    for ix, _ in enumerate(r.splitlines()):
        if not r.split()[cand_col] == '...':
            return False
    return True