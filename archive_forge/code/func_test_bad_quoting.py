import csv
from io import StringIO
import pytest
from pandas.compat import PY311
from pandas.errors import ParserError
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('quoting,msg', [('foo', '"quoting" must be an integer|Argument'), (10, 'bad "quoting" value')])
@xfail_pyarrow
def test_bad_quoting(all_parsers, quoting, msg):
    data = '1,2,3'
    parser = all_parsers
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), quoting=quoting)