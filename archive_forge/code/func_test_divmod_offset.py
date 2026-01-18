from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_divmod_offset(self):
    td = Timedelta(days=2, hours=6)
    result = divmod(td, offsets.Hour(-4))
    assert result[0] == -14
    assert isinstance(result[1], Timedelta)
    assert result[1] == Timedelta(hours=-2)