from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_divmod_invalid(self):
    td = Timedelta(days=2, hours=6)
    msg = "unsupported operand type\\(s\\) for //: 'Timedelta' and 'Timestamp'"
    with pytest.raises(TypeError, match=msg):
        divmod(td, Timestamp('2018-01-22'))