from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
@pytest.fixture
def planet_df():
    return DataFrame({'planet': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'], 'type': ['terrestrial', 'terrestrial', 'terrestrial', 'terrestrial', 'gas giant', 'gas giant', 'ice giant', 'ice giant'], 'location': ['inner', 'inner', 'inner', 'inner', 'outer', 'outer', 'outer', 'outer'], 'mass': [0.330114, 4.86747, 5.97237, 0.641712, 1898.187, 568.3174, 86.8127, 102.4126]})