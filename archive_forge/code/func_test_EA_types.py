import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason="data type 'json' not understood")
@pytest.mark.parametrize('engine', ['c', 'python'])
def test_EA_types(self, engine, data, request):
    super().test_EA_types(engine, data, request)