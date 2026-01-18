import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
@pytest.fixture
def left_multi():
    return DataFrame({'Origin': ['A', 'A', 'B', 'B', 'C'], 'Destination': ['A', 'B', 'A', 'C', 'A'], 'Period': ['AM', 'AM', 'IP', 'AM', 'OP'], 'TripPurp': ['hbw', 'nhb', 'hbo', 'nhb', 'hbw'], 'Trips': [1987, 3647, 2470, 4296, 4444]}, columns=['Origin', 'Destination', 'Period', 'TripPurp', 'Trips']).set_index(['Origin', 'Destination', 'Period', 'TripPurp'])