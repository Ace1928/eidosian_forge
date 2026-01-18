import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_mixed_float(self, mixed_float_frame):
    casted = mixed_float_frame.reindex(columns=['A', 'B']).astype('float32')
    _check_cast(casted, 'float32')
    casted = mixed_float_frame.reindex(columns=['A', 'B']).astype('float16')
    _check_cast(casted, 'float16')