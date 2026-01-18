from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def desc3(group):
    result = group.describe()
    result.index.name = f'stat_{len(group):d}'
    result = result[:len(group)]
    return result