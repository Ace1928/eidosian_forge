import csv
from io import StringIO
import pytest
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('kwargs,warning_klass', [({'sep': ','}, None), ({'sep': '.'}, ParserWarning), ({'delimiter': ':'}, None), ({'delimiter': None}, None), ({'delimiter': ','}, ParserWarning), ({'delimiter': '.'}, ParserWarning)], ids=['sep-override-true', 'sep-override-false', 'delimiter-no-conflict', 'delimiter-default-arg', 'delimiter-conflict', 'delimiter-conflict2'])
def test_dialect_conflict_delimiter(all_parsers, custom_dialect, kwargs, warning_klass):
    dialect_name, dialect_kwargs = custom_dialect
    parser = all_parsers
    expected = DataFrame({'a': [1], 'b': [2]})
    data = 'a:b\n1:2'
    with tm.with_csv_dialect(dialect_name, **dialect_kwargs):
        if parser.engine == 'pyarrow':
            msg = "The 'dialect' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv_check_warnings(None, "Conflicting values for 'delimiter'", StringIO(data), dialect=dialect_name, **kwargs)
            return
        result = parser.read_csv_check_warnings(warning_klass, "Conflicting values for 'delimiter'", StringIO(data), dialect=dialect_name, **kwargs)
        tm.assert_frame_equal(result, expected)