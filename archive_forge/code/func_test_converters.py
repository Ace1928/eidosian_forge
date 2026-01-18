from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('column', [3, 'D'])
@pytest.mark.parametrize('converter', [parse, lambda x: int(x.split('/')[2])])
def test_converters(all_parsers, column, converter):
    parser = all_parsers
    data = 'A,B,C,D\na,1,2,01/01/2009\nb,3,4,01/02/2009\nc,4,5,01/03/2009\n'
    if parser.engine == 'pyarrow':
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters={column: converter})
        return
    result = parser.read_csv(StringIO(data), converters={column: converter})
    expected = parser.read_csv(StringIO(data))
    expected['D'] = expected['D'].map(converter)
    tm.assert_frame_equal(result, expected)