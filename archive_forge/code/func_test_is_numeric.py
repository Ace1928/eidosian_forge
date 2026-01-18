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
@pytest.mark.parametrize('index, expected', [('string', False), ('bool-object', False), ('bool-dtype', False), ('categorical', False), ('int64', True), ('int32', True), ('uint64', True), ('uint32', True), ('datetime', False), ('float64', True), ('float32', True)], indirect=['index'])
def test_is_numeric(self, index, expected):
    assert is_any_real_numeric_dtype(index) is expected