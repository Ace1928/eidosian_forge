import numpy as np
import pytest
from pandas import (
def test_equals_multiindex(self):
    mi = MultiIndex.from_arrays([['A', 'B', 'C', 'D'], range(4)])
    ci = mi.to_flat_index().astype('category')
    assert not ci.equals(mi)