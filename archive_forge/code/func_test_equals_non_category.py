import numpy as np
import pytest
from pandas import (
def test_equals_non_category(self):
    ci = CategoricalIndex(['A', 'B', np.nan, np.nan])
    other = Index(['A', 'B', 'D', np.nan])
    assert not ci.equals(other)