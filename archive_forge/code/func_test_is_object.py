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
@pytest.mark.parametrize('index, expected', [('string', True), ('bool-object', True), ('bool-dtype', False), ('categorical', False), ('int64', False), ('int32', False), ('uint64', False), ('uint32', False), ('datetime', False), ('float64', False), ('float32', False)], indirect=['index'])
def test_is_object(self, index, expected, using_infer_string):
    if using_infer_string and index.dtype == 'string' and expected:
        expected = False
    assert is_object_dtype(index) is expected