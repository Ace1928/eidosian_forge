from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('suffixes', [{'left', 'right'}, {'left': 0, 'right': 0}])
def test_merge_suffix_raises(suffixes):
    a = DataFrame({'a': [1, 2, 3]})
    b = DataFrame({'b': [3, 4, 5]})
    with pytest.raises(TypeError, match="Passing 'suffixes' as a"):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)