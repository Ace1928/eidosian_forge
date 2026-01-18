from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_categorical_format():
    s = pd.Series(['a', 'b', 'c']).astype('category')
    known = dd.from_pandas(s, npartitions=1)
    unknown = known.cat.as_unknown()
    exp = 'Dask Series Structure:\nnpartitions=1\n0    category[known]\n2                ...\ndtype: category\nDask Name: from_pandas, 1 graph layer'
    assert repr(known) == exp
    exp = 'Dask Series Structure:\nnpartitions=1\n0    category[unknown]\n2                  ...\ndtype: category\nDask Name: from_pandas, 1 graph layer'
    assert repr(unknown) == exp