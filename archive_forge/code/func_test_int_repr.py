import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_int_repr():
    vec = robjects.vectors.IntVector((1, 2, ri.NA_Integer))
    s = repr(vec)
    assert s.endswith('[1, 2, NA_integer_]')