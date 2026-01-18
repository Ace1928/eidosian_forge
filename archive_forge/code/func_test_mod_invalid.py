from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_mod_invalid(self):
    td = Timedelta(hours=37)
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        td % Timestamp('2018-01-22')
    with pytest.raises(TypeError, match=msg):
        td % []