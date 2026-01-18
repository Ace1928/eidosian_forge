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
@pytest.mark.parametrize('dt_conv, arg', [(pd.to_datetime, ['2000-01-01', '2000-01-02']), (pd.to_timedelta, ['01:02:03', '01:02:04'])])
def test_dt_conversion_preserves_name(self, dt_conv, arg):
    index = Index(arg, name='label')
    assert index.name == dt_conv(index).name