from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('opname', ['eq', 'ne', 'le', 'lt', 'ge', 'gt', 'add', 'radd', 'sub', 'rsub', 'mul', 'rmul', 'truediv', 'rtruediv', 'floordiv', 'rfloordiv', 'pow', 'rpow', 'mod', 'divmod'])
def test_generated_op_names(opname, index):
    opname = f'__{opname}__'
    method = getattr(index, opname)
    assert method.__name__ == opname