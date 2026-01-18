from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
def test_groupby_blocklist(df_letters):
    df = df_letters
    s = df_letters.floats
    blocklist = ['eval', 'query', 'abs', 'where', 'mask', 'align', 'groupby', 'clip', 'astype', 'at', 'combine', 'consolidate', 'convert_objects']
    to_methods = [method for method in dir(df) if method.startswith('to_')]
    blocklist.extend(to_methods)
    for bl in blocklist:
        for obj in (df, s):
            gb = obj.groupby(df.letters)
            defined_but_not_allowed = f"(?:^Cannot.+{repr(bl)}.+'{type(gb).__name__}'.+try using the 'apply' method$)"
            not_defined = f"(?:^'{type(gb).__name__}' object has no attribute {repr(bl)}$)"
            msg = f'{defined_but_not_allowed}|{not_defined}'
            with pytest.raises(AttributeError, match=msg):
                getattr(gb, bl)