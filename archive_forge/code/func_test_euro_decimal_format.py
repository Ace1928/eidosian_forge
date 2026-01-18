from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_euro_decimal_format(all_parsers):
    parser = all_parsers
    data = 'Id;Number1;Number2;Text1;Text2;Number3\n1;1521,1541;187101,9543;ABC;poi;4,738797819\n2;121,12;14897,76;DEF;uyt;0,377320872\n3;878,158;108013,434;GHI;rez;2,735694704'
    result = parser.read_csv(StringIO(data), sep=';', decimal=',')
    expected = DataFrame([[1, 1521.1541, 187101.9543, 'ABC', 'poi', 4.738797819], [2, 121.12, 14897.76, 'DEF', 'uyt', 0.377320872], [3, 878.158, 108013.434, 'GHI', 'rez', 2.735694704]], columns=['Id', 'Number1', 'Number2', 'Text1', 'Text2', 'Number3'])
    tm.assert_frame_equal(result, expected)