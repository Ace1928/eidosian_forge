from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouping_is_iterable(self, tsframe):
    grouped = tsframe.groupby([lambda x: x.weekday(), lambda x: x.year])
    for g in grouped._grouper.groupings[0]:
        pass