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
@pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
def test_get_handle_with_path(self, path_type):
    with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
        filename = path_type('~/' + Path(tmp).name + '/sometest')
        with icom.get_handle(filename, 'w') as handles:
            assert Path(handles.handle.name).is_absolute()
            assert os.path.expanduser(filename) == handles.handle.name