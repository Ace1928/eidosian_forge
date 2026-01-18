from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_constructor_invalid_args(self):
    msg = 'RangeIndex\\(\\.\\.\\.\\) must be called with integers'
    with pytest.raises(TypeError, match=msg):
        RangeIndex()
    with pytest.raises(TypeError, match=msg):
        RangeIndex(name='Foo')
    msg = 'Index\\(\\.\\.\\.\\) must be called with a collection of some kind, 0 was passed'
    with pytest.raises(TypeError, match=msg):
        Index(0)