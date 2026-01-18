import numpy as np
import pytest
from pandas import (
from pandas.io.formats.format import EngFormatter
def test_eng_float_formatter2(self, float_frame):
    df = float_frame
    df.loc[5] = 0
    set_eng_float_format()
    repr(df)
    set_eng_float_format(use_eng_prefix=True)
    repr(df)
    set_eng_float_format(accuracy=0)
    repr(df)