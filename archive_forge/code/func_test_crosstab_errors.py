import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_errors(self):
    df = DataFrame({'a': [1, 2, 2, 2, 2], 'b': [3, 3, 4, 4, 4], 'c': [1, 1, np.nan, 1, 1]})
    error = 'values cannot be used without an aggfunc.'
    with pytest.raises(ValueError, match=error):
        crosstab(df.a, df.b, values=df.c)
    error = 'aggfunc cannot be used without values'
    with pytest.raises(ValueError, match=error):
        crosstab(df.a, df.b, aggfunc=np.mean)
    error = 'Not a valid normalize argument'
    with pytest.raises(ValueError, match=error):
        crosstab(df.a, df.b, normalize='42')
    with pytest.raises(ValueError, match=error):
        crosstab(df.a, df.b, normalize=42)
    error = 'Not a valid margins argument'
    with pytest.raises(ValueError, match=error):
        crosstab(df.a, df.b, normalize='all', margins=42)