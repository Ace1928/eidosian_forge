from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def check_fun(self, testfunc, targfunc, testar, skipna, empty_targfunc=None, **kwargs):
    targar = testar
    if testar.endswith('_nan') and hasattr(self, testar[:-4]):
        targar = testar[:-4]
    testarval = getattr(self, testar)
    targarval = getattr(self, targar)
    self.check_fun_data(testfunc, targfunc, testarval, targarval, skipna=skipna, empty_targfunc=empty_targfunc, **kwargs)