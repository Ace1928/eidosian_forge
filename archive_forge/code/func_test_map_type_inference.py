from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_type_inference():
    s = Series(range(3))
    s2 = s.map(lambda x: np.where(x == 0, 0, 1))
    assert issubclass(s2.dtype.type, np.integer)