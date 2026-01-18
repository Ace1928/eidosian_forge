import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_not_read_only(self):
    ser = Series([1, 2], dtype=object)
    with pd.option_context('mode.copy_on_write', True):
        idx = Index(ser)
        assert idx._values.flags.writeable