from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arg', [[1], 1, 'True', [], 0])
def test_drop_duplicates_non_boolean_ignore_index(arg):
    df = DataFrame({'a': [1, 2, 1, 3]})
    msg = '^For argument "ignore_index" expected type bool, received type .*.$'
    with pytest.raises(ValueError, match=msg):
        df.drop_duplicates(ignore_index=arg)