from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('usecol, engine', [([0, 1, 1], 'python'), ([0, 1, 1], 'c')])
def test_mangle_cols_names(all_parsers, usecol, engine):
    parser = all_parsers
    data = '1,2,3'
    names = ['A', 'A', 'B']
    with pytest.raises(ValueError, match='Duplicate names'):
        parser.read_csv(StringIO(data), names=names, usecols=usecol, engine=engine)