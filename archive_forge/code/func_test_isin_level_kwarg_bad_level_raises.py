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
def test_isin_level_kwarg_bad_level_raises(self, index):
    for level in [10, index.nlevels, -(index.nlevels + 1)]:
        with pytest.raises(IndexError, match='Too many levels'):
            index.isin([], level=level)