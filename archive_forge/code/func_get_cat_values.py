import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def get_cat_values(ndframe):
    return ndframe._mgr.arrays[0]