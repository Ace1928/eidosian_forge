from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_names', [['', ''], ['foo', ''], ['', 'bar'], ['foo', 'bar'], ['NotReallyUnnamed', 'Unnamed: 0']])
def test_multi_index_naming(all_parsers, index_names, request):
    parser = all_parsers
    if parser.engine == 'pyarrow' and '' in index_names:
        mark = pytest.mark.xfail(reason='One case raises, others are wrong')
        request.applymarker(mark)
    data = ','.join(index_names + ['col\na,c,1\na,d,2\nb,c,3\nb,d,4'])
    result = parser.read_csv(StringIO(data), index_col=[0, 1])
    expected = DataFrame({'col': [1, 2, 3, 4]}, index=MultiIndex.from_product([['a', 'b'], ['c', 'd']]))
    expected.index.names = [name if name else None for name in index_names]
    tm.assert_frame_equal(result, expected)