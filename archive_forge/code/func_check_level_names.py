import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def check_level_names(index, names):
    assert [level.name for level in index.levels] == list(names)