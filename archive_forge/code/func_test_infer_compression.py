import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('ext', [None, 'gz', 'bz2'])
def test_infer_compression(all_parsers, csv1, buffer, ext):
    parser = all_parsers
    kwargs = {'index_col': 0, 'parse_dates': True}
    expected = parser.read_csv(csv1, **kwargs)
    kwargs['compression'] = 'infer'
    if buffer:
        with open(csv1, encoding='utf-8') as f:
            result = parser.read_csv(f, **kwargs)
    else:
        ext = '.' + ext if ext else ''
        result = parser.read_csv(csv1 + ext, **kwargs)
    tm.assert_frame_equal(result, expected)