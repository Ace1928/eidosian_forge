from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def f_constant_df(group):
    names.append(group.name)
    return DataFrame({'a': [1], 'b': [1]})