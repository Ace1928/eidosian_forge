from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_consolidate_inplace(self, float_frame):
    for letter in range(ord('A'), ord('Z')):
        float_frame[chr(letter)] = chr(letter)