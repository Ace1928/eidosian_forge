import re
import numpy as np
import pytest
from pandas import (
def test_delitem_corner(self, float_frame):
    f = float_frame.copy()
    del f['D']
    assert len(f.columns) == 3
    with pytest.raises(KeyError, match="^'D'$"):
        del f['D']
    del f['B']
    assert len(f.columns) == 2