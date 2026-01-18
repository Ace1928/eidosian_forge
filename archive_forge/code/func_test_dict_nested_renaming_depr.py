from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['agg', 'transform'])
def test_dict_nested_renaming_depr(method):
    df = DataFrame({'A': range(5), 'B': 5})
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        getattr(df, method)({'A': {'foo': 'min'}, 'B': {'bar': 'max'}})