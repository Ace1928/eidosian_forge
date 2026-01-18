import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_method(self):
    s_empty = Series(dtype=object)
    assert s_empty.empty