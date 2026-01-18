from copy import deepcopy
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_deepcopy_empty(self):
    empty_frame = DataFrame(data=[], index=[], columns=['A'])
    empty_frame_copy = deepcopy(empty_frame)
    tm.assert_frame_equal(empty_frame_copy, empty_frame)