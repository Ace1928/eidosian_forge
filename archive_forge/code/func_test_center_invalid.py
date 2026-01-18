import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_center_invalid():
    df = DataFrame()
    with pytest.raises(TypeError, match='.* got an unexpected keyword'):
        df.expanding(center=True)