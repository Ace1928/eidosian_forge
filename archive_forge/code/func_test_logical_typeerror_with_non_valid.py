from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
@pytest.mark.parametrize('op,res', [('__eq__', False), ('__ne__', True)])
@pytest.mark.filterwarnings('ignore:elementwise:FutureWarning')
def test_logical_typeerror_with_non_valid(self, op, res, float_frame):
    result = getattr(float_frame, op)('foo')
    assert bool(result.all().all()) is res