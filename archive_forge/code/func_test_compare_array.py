import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
def test_compare_array(self, data, comparison_op, request):
    if comparison_op.__name__ in ['eq', 'ne']:
        mark = pytest.mark.xfail(reason='Comparison methods not implemented')
        request.applymarker(mark)
    super().test_compare_array(data, comparison_op)