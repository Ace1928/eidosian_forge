import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_from_list_no_dtype(self):
    index = Index([1, 2, 3])
    assert index.dtype == np.int64