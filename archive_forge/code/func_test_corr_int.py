import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corr_int(self):
    df = DataFrame({'a': [1, 2, 3, 4], 'b': [1, 2, 3, 4]})
    df.cov()
    df.corr()