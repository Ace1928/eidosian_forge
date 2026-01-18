from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('kind', ('box', 'scatter', 'hexbin'))
def test_group_subplot_invalid_kind(self, kind):
    d = {'a': np.arange(10), 'b': np.arange(10)}
    df = DataFrame(d)
    with pytest.raises(ValueError, match='When subplots is an iterable, kind must be one of'):
        df.plot(subplots=[('a', 'b')], kind=kind)