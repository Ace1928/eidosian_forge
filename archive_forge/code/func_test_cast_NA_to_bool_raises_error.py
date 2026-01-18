from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('data, na_values', [('false,1\n,1\ntrue', None), ('false,1\nnull,1\ntrue', None), ('false,1\nnan,1\ntrue', None), ('false,1\nfoo,1\ntrue', 'foo'), ('false,1\nfoo,1\ntrue', ['foo']), ('false,1\nfoo,1\ntrue', {'a': 'foo'})])
def test_cast_NA_to_bool_raises_error(all_parsers, data, na_values):
    parser = all_parsers
    msg = '|'.join(['Bool column has NA values in column [0a]', 'cannot safely convert passed user dtype of bool for object dtyped data in column 0'])
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=None, names=['a', 'b'], dtype={'a': 'bool'}, na_values=na_values)