from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_name_printing(self):
    s = Series(index=date_range('20010101', '20020101'), name='test', dtype=object)
    assert 'Name: test' in repr(s)