from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_compare_tick(self, tick_classes):
    cls = tick_classes
    off = cls(4)
    td = off._as_pd_timedelta
    assert isinstance(td, Timedelta)
    assert td == off
    assert not td != off
    assert td <= off
    assert td >= off
    assert not td < off
    assert not td > off
    assert not td == 2 * off
    assert td != 2 * off
    assert td <= 2 * off
    assert td < 2 * off
    assert not td >= 2 * off
    assert not td > 2 * off