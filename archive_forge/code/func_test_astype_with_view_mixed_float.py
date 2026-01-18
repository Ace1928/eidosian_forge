import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_with_view_mixed_float(self, mixed_float_frame):
    tf = mixed_float_frame.reindex(columns=['A', 'B', 'C'])
    tf.astype(np.int64)
    tf.astype(np.float32)