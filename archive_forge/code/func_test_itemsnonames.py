import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_itemsnonames():
    vec = robjects.IntVector(range(3))
    names = [k for k, v in vec.items()]
    assert names == [None, None, None]
    values = [v for k, v in vec.items()]
    assert values == [0, 1, 2]