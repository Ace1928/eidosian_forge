from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('with_header', [True, False])
def test_index_col_named(all_parsers, with_header):
    parser = all_parsers
    no_header = 'KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000\nKORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000\nKORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000\nKORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000\nKORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000\nKORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000'
    header = 'ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir\n'
    if with_header:
        data = header + no_header
        result = parser.read_csv(StringIO(data), index_col='ID')
        expected = parser.read_csv(StringIO(data), header=0).set_index('ID')
        tm.assert_frame_equal(result, expected)
    else:
        data = no_header
        msg = 'Index ID invalid'
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), index_col='ID')