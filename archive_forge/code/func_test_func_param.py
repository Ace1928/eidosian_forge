import json
import numpy as np
import pandas as pd
from pytest import raises
from scipy import stats
from triad import to_uuid
from tune._utils import assert_close
from tune.concepts.space import (
def test_func_param():

    def tf(*args, x, y):
        return sum(args) + x + y
    f1 = FuncParam(tf, 4, x=1, y=2)
    assert 7 == f1()
    f2 = FuncParam(tf, 4, x=1, y=2)
    f3 = FuncParam(tf, 5, x=1, y=2)
    assert f1 == f2
    assert f1 != f3
    assert to_uuid(f1) == to_uuid(f2)
    assert to_uuid(f1) != to_uuid(f3)
    f1[0] = 5
    f1['y'] = 3
    assert 5 == f1[0]
    assert 3 == f1['y']
    assert 9 == f1()