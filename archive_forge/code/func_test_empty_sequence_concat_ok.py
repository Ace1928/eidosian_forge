import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arg', [[DataFrame()], [None, DataFrame()], [DataFrame(), None]])
def test_empty_sequence_concat_ok(self, arg):
    pd.concat(arg)