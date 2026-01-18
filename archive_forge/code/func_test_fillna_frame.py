import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='Setting a dict as a scalar')
def test_fillna_frame(self):
    """We treat dictionaries as a mapping in fillna, not a scalar."""
    super().test_fillna_frame()