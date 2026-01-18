from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
@pytest.mark.parametrize('memory_map', [True, False])
def test_encoding_mmap(memory_map):
    """
    encoding should be working, even when using a memory-mapped file.

    GH 23254.
    """
    encoding = 'iso8859_1'
    with tm.ensure_clean() as path:
        Path(path).write_bytes(' 1 A Ä 2\n'.encode(encoding))
        df = read_fwf(path, header=None, widths=[2, 2, 2, 2], encoding=encoding, memory_map=memory_map)
    df_reference = DataFrame([[1, 'A', 'Ä', 2]])
    tm.assert_frame_equal(df, df_reference)