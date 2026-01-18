import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_can_hold_identifiers(self, simple_index):
    idx = simple_index
    key = idx[0]
    assert idx._can_hold_identifiers_and_holds_name(key) is False