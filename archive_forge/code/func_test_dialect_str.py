import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
def test_dialect_str(all_parsers):
    dialect_name = 'mydialect'
    parser = all_parsers
    data = 'fruit:vegetable\napple:broccoli\npear:tomato\n'
    exp = DataFrame({'fruit': ['apple', 'pear'], 'vegetable': ['broccoli', 'tomato']})
    with tm.with_csv_dialect(dialect_name, delimiter=':'):
        if parser.engine == 'pyarrow':
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), dialect=dialect_name)
            return
        df = parser.read_csv(StringIO(data), dialect=dialect_name)
        tm.assert_frame_equal(df, exp)