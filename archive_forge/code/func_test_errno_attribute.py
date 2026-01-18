import codecs
import errno
from functools import partial
from io import (
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
def test_errno_attribute():
    with pytest.raises(FileNotFoundError, match='\\[Errno 2\\]') as err:
        pd.read_csv('doesnt_exist')
        assert err.errno == errno.ENOENT