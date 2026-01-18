from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_top_level(self):
    midx = MultiIndex.from_product([['A', 'B'], [1, 2]])
    assert 'A' in midx
    assert 'A' not in midx._engine