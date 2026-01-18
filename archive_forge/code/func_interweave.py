import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def interweave(list_obj):
    temp = []
    for x in list_obj:
        temp.extend([x, x])
    return temp