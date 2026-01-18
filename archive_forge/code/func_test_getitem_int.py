import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
@pytest.mark.parametrize('cls,values', [(robjects.StrVector, ['abc', 'def']), (robjects.IntVector, [123, 456]), (robjects.FloatVector, [123.0, 456.0]), (robjects.BoolVector, [True, False])])
def test_getitem_int(cls, values):
    vec = cls(values)
    assert vec[0] == values[0]
    assert vec[1] == values[1]