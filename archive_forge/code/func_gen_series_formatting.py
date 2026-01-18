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
def gen_series_formatting():
    s1 = Series(['a'] * 100)
    s2 = Series(['ab'] * 100)
    s3 = Series(['a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef'])
    s4 = s3[::-1]
    test_sers = {'onel': s1, 'twol': s2, 'asc': s3, 'desc': s4}
    return test_sers