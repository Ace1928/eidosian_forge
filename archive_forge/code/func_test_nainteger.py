import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_nainteger():
    vec = robjects.IntVector(range(3))
    vec[0] = robjects.NA_Integer
    assert robjects.baseenv['is.na'](vec)[0] is True