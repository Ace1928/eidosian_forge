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
def test_internal_eof_byte_to_file(all_parsers):
    parser = all_parsers
    data = b'c1,c2\r\n"test \x1a    test", test\r\n'
    expected = DataFrame([['test \x1a    test', ' test']], columns=['c1', 'c2'])
    path = f'__{uuid.uuid4()}__.csv'
    with tm.ensure_clean(path) as path:
        with open(path, 'wb') as f:
            f.write(data)
        result = parser.read_csv(path)
        tm.assert_frame_equal(result, expected)