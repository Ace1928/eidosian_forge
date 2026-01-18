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
def test_close_on_error():

    class TestError:

        def close(self):
            raise OSError('test')
    with pytest.raises(OSError, match='test'):
        with BytesIO() as buffer:
            with icom.get_handle(buffer, 'rb') as handles:
                handles.created_handles.append(TestError())