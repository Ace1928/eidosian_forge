from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('conv_f', [lambda x: x, str])
def test_converter_index_col_bug(all_parsers, conv_f):
    parser = all_parsers
    data = 'A;B\n1;2\n3;4'
    if parser.engine == 'pyarrow':
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=';', index_col='A', converters={'A': conv_f})
        return
    rs = parser.read_csv(StringIO(data), sep=';', index_col='A', converters={'A': conv_f})
    xp = DataFrame({'B': [2, 4]}, index=Index(['1', '3'], name='A', dtype='object'))
    tm.assert_frame_equal(rs, xp)