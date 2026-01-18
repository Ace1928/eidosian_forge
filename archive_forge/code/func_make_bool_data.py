import warnings
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import (
from pandas.core.arrays.integer import (
from pandas.tests.extension import base
def make_bool_data():
    return [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False]