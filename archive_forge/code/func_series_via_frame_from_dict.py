from datetime import datetime
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
def series_via_frame_from_dict(x, **kwargs):
    return DataFrame({'a': x}, **kwargs)['a']