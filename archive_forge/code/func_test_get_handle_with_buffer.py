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
def test_get_handle_with_buffer(self):
    with StringIO() as input_buffer:
        with icom.get_handle(input_buffer, 'r') as handles:
            assert handles.handle == input_buffer
        assert not input_buffer.closed
    assert input_buffer.closed