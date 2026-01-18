import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
def test_excessively_long_string(self):
    str_lens = (1, 244, 500)
    s = {}
    for str_len in str_lens:
        s['s' + str(str_len)] = Series(['a' * str_len, 'b' * str_len, 'c' * str_len])
    original = DataFrame(s)
    msg = "Fixed width strings in Stata \\.dta files are limited to 244 \\(or fewer\\)\\ncharacters\\.  Column 's500' does not satisfy this restriction\\. Use the\\n'version=117' parameter to write the newer \\(Stata 13 and later\\) format\\."
    with pytest.raises(ValueError, match=msg):
        with tm.ensure_clean() as path:
            original.to_stata(path)