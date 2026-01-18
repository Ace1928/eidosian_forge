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
def test_bytesiowrapper_returns_correct_bytes(self):
    data = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
    with icom.get_handle(StringIO(data), 'rb', is_text=False) as handles:
        result = b''
        chunksize = 5
        while True:
            chunk = handles.handle.read(chunksize)
            assert len(chunk) <= chunksize
            if len(chunk) < chunksize:
                assert len(handles.handle.read()) == 0
                result += chunk
                break
            result += chunk
        assert result == data.encode('utf-8')