from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_memory_map(all_parsers, csv_dir_path):
    mmap_file = os.path.join(csv_dir_path, 'test_mmap.csv')
    parser = all_parsers
    expected = DataFrame({'a': [1, 2, 3], 'b': ['one', 'two', 'three'], 'c': ['I', 'II', 'III']})
    if parser.engine == 'pyarrow':
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(mmap_file, memory_map=True)
        return
    result = parser.read_csv(mmap_file, memory_map=True)
    tm.assert_frame_equal(result, expected)