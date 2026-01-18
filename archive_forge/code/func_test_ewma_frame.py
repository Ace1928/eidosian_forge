import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name', ['var', 'std', 'mean'])
def test_ewma_frame(frame, name):
    frame_result = getattr(frame.ewm(com=10), name)()
    assert isinstance(frame_result, DataFrame)