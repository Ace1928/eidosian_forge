import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
def test_invalid_dialect(all_parsers):

    class InvalidDialect:
        pass
    data = 'a\n1'
    parser = all_parsers
    msg = 'Invalid dialect'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dialect=InvalidDialect)