from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_min_max_categorical(self):
    ci = pd.CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=False)
    msg = 'Categorical is not ordered for operation min\\nyou can use .as_ordered\\(\\) to change the Categorical to an ordered one\\n'
    with pytest.raises(TypeError, match=msg):
        ci.min()
    msg = 'Categorical is not ordered for operation max\\nyou can use .as_ordered\\(\\) to change the Categorical to an ordered one\\n'
    with pytest.raises(TypeError, match=msg):
        ci.max()
    ci = pd.CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=True)
    assert ci.min() == 'c'
    assert ci.max() == 'b'