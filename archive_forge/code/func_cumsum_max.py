import numpy as np
import pytest
from pandas._libs import lib
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def cumsum_max(x):
    x.cumsum().max()
    return 0