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
def has_vertically_truncated_repr(df):
    r = repr(df)
    only_dot_row = False
    for row in r.splitlines():
        if re.match('^[\\.\\ ]+$', row):
            only_dot_row = True
    return only_dot_row