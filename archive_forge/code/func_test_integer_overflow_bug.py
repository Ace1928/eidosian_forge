from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('sep', [' ', '\\s+'])
def test_integer_overflow_bug(all_parsers, sep):
    data = '65248E10 11\n55555E55 22\n'
    parser = all_parsers
    if parser.engine == 'pyarrow' and sep != ' ':
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, sep=sep)
        return
    result = parser.read_csv(StringIO(data), header=None, sep=sep)
    expected = DataFrame([[652480000000000.0, 11], [5.5555e+59, 22]])
    tm.assert_frame_equal(result, expected)