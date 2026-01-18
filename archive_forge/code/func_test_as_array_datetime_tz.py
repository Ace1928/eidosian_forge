from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
def test_as_array_datetime_tz(self):
    mgr = create_mgr('h: M8[ns, US/Eastern]; g: M8[ns, CET]')
    assert mgr.iget(0).dtype == 'datetime64[ns, US/Eastern]'
    assert mgr.iget(1).dtype == 'datetime64[ns, CET]'
    assert mgr.as_array().dtype == 'object'