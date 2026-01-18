import re
import sys
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('subset', ['a', ['a'], ['a', 'B']])
def test_duplicated_with_misspelled_column_name(subset):
    df = DataFrame({'A': [0, 0, 1], 'B': [0, 0, 1], 'C': [0, 0, 1]})
    msg = re.escape("Index(['a'], dtype=")
    with pytest.raises(KeyError, match=msg):
        df.duplicated(subset)