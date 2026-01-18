import numpy as np
import pytest
from pandas import Index
import pandas._testing as tm
def test_delete_raises(self):
    index = Index(['a', 'b', 'c', 'd'], name='index')
    msg = 'index 5 is out of bounds for axis 0 with size 4'
    with pytest.raises(IndexError, match=msg):
        index.delete(5)