from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('pass_encoding', [True, False])
def test_encoding_temp_file(all_parsers, utf_value, encoding_fmt, pass_encoding):
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)
    if parser.engine == 'pyarrow' and pass_encoding is True and (utf_value in [16, 32]):
        pytest.skip('These cases freeze')
    expected = DataFrame({'foo': ['bar']})
    with tm.ensure_clean(mode='w+', encoding=encoding, return_filelike=True) as f:
        f.write('foo\nbar')
        f.seek(0)
        result = parser.read_csv(f, encoding=encoding if pass_encoding else None)
        tm.assert_frame_equal(result, expected)