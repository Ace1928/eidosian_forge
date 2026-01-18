import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_list_of_dataclasses_error_thrown(self):
    Point = make_dataclass('Point', [('x', int), ('y', int)])
    msg = 'asdict() should be called on dataclass instances'
    with pytest.raises(TypeError, match=re.escape(msg)):
        DataFrame([Point(0, 0), {'x': 1, 'y': 0}])