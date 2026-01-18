from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_buffer_rd_bytes_bad_unicode(c_parser_only):
    t = BytesIO(b'\xb0')
    t = TextIOWrapper(t, encoding='ascii', errors='surrogateescape')
    msg = "'utf-8' codec can't encode character"
    with pytest.raises(UnicodeError, match=msg):
        c_parser_only.read_csv(t, encoding='UTF-8')