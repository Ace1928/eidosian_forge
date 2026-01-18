import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_float_repr():
    vec = robjects.vectors.FloatVector((1, 2, 3))
    r = repr(vec).split('\n')
    assert r[-1].startswith('[')
    assert r[-1].endswith(']')
    assert len(r[-1].split(',')) == 3