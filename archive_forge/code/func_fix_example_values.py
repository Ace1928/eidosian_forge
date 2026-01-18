import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def fix_example_values(actual_cols, expected_cols):
    """
    Fix type of expected values (as read from JSON) according to
    actual ORC datatype.
    """
    for name in expected_cols:
        expected = expected_cols[name]
        actual = actual_cols[name]
        if name == 'map' and [d.keys() == {'key', 'value'} for m in expected for d in m]:
            col = expected_cols[name].copy()
            for i, m in enumerate(expected):
                col[i] = [(d['key'], d['value']) for d in m]
            expected_cols[name] = col
            continue
        typ = actual[0].__class__
        if issubclass(typ, datetime.datetime):
            expected = pd.to_datetime(expected)
        elif issubclass(typ, datetime.date):
            expected = expected.dt.date
        elif typ is decimal.Decimal:
            converted_decimals = [None] * len(expected)
            for i, (d, v) in enumerate(zip(actual, expected)):
                if not pd.isnull(v):
                    exp = d.as_tuple().exponent
                    factor = 10 ** (-exp)
                    converted_decimals[i] = decimal.Decimal(round(v * factor)).scaleb(exp)
            expected = pd.Series(converted_decimals)
        expected_cols[name] = expected