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
def has_expanded_repr(df):
    r = repr(df)
    for line in r.split('\n'):
        if line.endswith('\\'):
            return True
    return False