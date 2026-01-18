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
def test_int_name_format(self, frame_or_series):
    index = Index(['a', 'b', 'c'], name=0)
    result = frame_or_series(list(range(3)), index=index)
    assert '0' in repr(result)