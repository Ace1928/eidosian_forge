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
@pytest.mark.parametrize('attr', ['is_monotonic_increasing', 'is_monotonic_decreasing', '_is_strictly_monotonic_increasing', '_is_strictly_monotonic_decreasing'])
def test_is_monotonic_incomparable(self, attr):
    index = Index([5, datetime.now(), 7])
    assert not getattr(index, attr)