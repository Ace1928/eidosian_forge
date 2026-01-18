from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_str(self):
    df = random_text(100000)
    indexer = df.letters.apply(lambda x: len(x) > 10)
    df.loc[indexer, 'letters'] = df.loc[indexer, 'letters'].apply(str.lower)