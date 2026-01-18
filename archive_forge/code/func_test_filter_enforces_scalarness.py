from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_enforces_scalarness():
    df = DataFrame([['best', 'a', 'x'], ['worst', 'b', 'y'], ['best', 'c', 'x'], ['best', 'd', 'y'], ['worst', 'd', 'y'], ['worst', 'd', 'y'], ['best', 'd', 'z']], columns=['a', 'b', 'c'])
    with pytest.raises(TypeError, match='filter function returned a.*'):
        df.groupby('c').filter(lambda g: g['a'] == 'best')