import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_get_standard_colors_consistency(self):
    from pandas.plotting._matplotlib.style import get_standard_colors
    color1 = get_standard_colors(1, color_type='random')
    color2 = get_standard_colors(1, color_type='random')
    assert color1 == color2