import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import df_equals, test_data_values
def test_crosstab():
    a = np.array(['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'], dtype=object)
    b = np.array(['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'], dtype=object)
    c = np.array(['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'], dtype=object)
    with warns_that_defaulting_to_pandas():
        df = pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
        assert isinstance(df, pd.DataFrame)
    foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
    with warns_that_defaulting_to_pandas():
        df = pd.crosstab(foo, bar)
        assert isinstance(df, pd.DataFrame)
    with warns_that_defaulting_to_pandas():
        df = pd.crosstab(foo, bar, dropna=False)
        assert isinstance(df, pd.DataFrame)