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
def test_get_handle_pyarrow_compat(self):
    pa_csv = pytest.importorskip('pyarrow.csv')
    data = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
    expected = pd.DataFrame({'a': ['1', '©', 'Look'], 'b': ['2', '®', 'a snake'], 'c': ['3', '®', '🐍']})
    s = StringIO(data)
    with icom.get_handle(s, 'rb', is_text=False) as handles:
        df = pa_csv.read_csv(handles.handle).to_pandas()
        tm.assert_frame_equal(df, expected)
        assert not s.closed