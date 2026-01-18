from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_casemethods(any_string_dtype):
    values = ['aaa', 'bbb', 'CCC', 'Dddd', 'eEEE']
    s = Series(values, dtype=any_string_dtype)
    assert s.str.lower().tolist() == [v.lower() for v in values]
    assert s.str.upper().tolist() == [v.upper() for v in values]
    assert s.str.title().tolist() == [v.title() for v in values]
    assert s.str.capitalize().tolist() == [v.capitalize() for v in values]
    assert s.str.swapcase().tolist() == [v.swapcase() for v in values]