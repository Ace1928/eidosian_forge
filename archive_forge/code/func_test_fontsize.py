import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_fontsize(self):
    df = DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [0, 0, 0, 1, 1, 1]})
    _check_ticks_props(df.boxplot('a', by='b', fontsize=16), xlabelsize=16, ylabelsize=16)