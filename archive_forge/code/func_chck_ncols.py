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
def chck_ncols(self, s):
    lines = [line for line in repr(s).split('\n') if not re.match('[^\\.]*\\.+', line)][:-1]
    ncolsizes = len({len(line.strip()) for line in lines})
    assert ncolsizes == 1