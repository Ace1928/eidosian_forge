from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_converters_euro_decimal_format(all_parsers):
    converters = {}
    parser = all_parsers
    data = 'Id;Number1;Number2;Text1;Text2;Number3\n1;1521,1541;187101,9543;ABC;poi;4,7387\n2;121,12;14897,76;DEF;uyt;0,3773\n3;878,158;108013,434;GHI;rez;2,7356'
    converters['Number1'] = converters['Number2'] = converters['Number3'] = lambda x: float(x.replace(',', '.'))
    if parser.engine == 'pyarrow':
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=';', converters=converters)
        return
    result = parser.read_csv(StringIO(data), sep=';', converters=converters)
    expected = DataFrame([[1, 1521.1541, 187101.9543, 'ABC', 'poi', 4.7387], [2, 121.12, 14897.76, 'DEF', 'uyt', 0.3773], [3, 878.158, 108013.434, 'GHI', 'rez', 2.7356]], columns=['Id', 'Number1', 'Number2', 'Text1', 'Text2', 'Number3'])
    tm.assert_frame_equal(result, expected)