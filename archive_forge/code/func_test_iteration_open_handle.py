from io import StringIO
import pytest
from pandas import (
import pandas._testing as tm
def test_iteration_open_handle(all_parsers):
    parser = all_parsers
    kwargs = {'header': None}
    with tm.ensure_clean() as path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG')
        with open(path, encoding='utf-8') as f:
            for line in f:
                if 'CCC' in line:
                    break
            result = parser.read_csv(f, **kwargs)
            expected = DataFrame({0: ['DDD', 'EEE', 'FFF', 'GGG']})
            tm.assert_frame_equal(result, expected)