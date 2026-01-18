from __future__ import annotations
from array import array
import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.compat.compressors import flatten_buffer
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data
import pandas.io.common as icom
from pandas.tseries.offsets import (
@pytest.mark.parametrize('data', [b'123', b'123456', bytearray(b'123'), memoryview(b'123'), pickle.PickleBuffer(b'123'), array('I', [1, 2, 3]), memoryview(b'123456').cast('B', (3, 2)), memoryview(b'123456').cast('B', (3, 2))[::2], np.arange(12).reshape((3, 4), order='C'), np.arange(12).reshape((3, 4), order='F'), np.arange(12).reshape((3, 4), order='C')[:, ::2]])
def test_flatten_buffer(data):
    result = flatten_buffer(data)
    expected = memoryview(data).tobytes('A')
    assert result == expected
    if isinstance(data, (bytes, bytearray)):
        assert result is data
    elif isinstance(result, memoryview):
        assert result.ndim == 1
        assert result.format == 'B'
        assert result.contiguous
        assert result.shape == (result.nbytes,)