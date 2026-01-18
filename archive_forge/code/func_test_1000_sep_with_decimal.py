from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('data,thousands,decimal', [('A|B|C\n1|2,334.01|5\n10|13|10.\n', ',', '.'), ('A|B|C\n1|2.334,01|5\n10|13|10,\n', '.', ',')])
def test_1000_sep_with_decimal(all_parsers, data, thousands, decimal):
    parser = all_parsers
    expected = DataFrame({'A': [1, 10], 'B': [2334.01, 13], 'C': [5, 10.0]})
    if parser.engine == 'pyarrow':
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep='|', thousands=thousands, decimal=decimal)
        return
    result = parser.read_csv(StringIO(data), sep='|', thousands=thousands, decimal=decimal)
    tm.assert_frame_equal(result, expected)