import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_floating_index_doc_example(self):
    index = Index([1.5, 2, 3, 4.5, 5])
    s = Series(range(5), index=index)
    assert s[3] == 2
    assert s.loc[3] == 2
    assert s.iloc[3] == 3