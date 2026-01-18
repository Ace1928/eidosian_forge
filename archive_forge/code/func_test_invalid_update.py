import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_invalid_update(self):
    df = DataFrame({'a': range(5), 'b': range(5)})
    online_ewm = df.head(2).ewm(0.5).online()
    with pytest.raises(ValueError, match='Must call mean with update=None first before passing update'):
        online_ewm.mean(update=df.head(1))