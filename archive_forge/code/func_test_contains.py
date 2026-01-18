import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_contains():
    v = robjects.StrVector(('abc', 'def', 'ghi'))
    assert 'def' in v.ro
    assert 'foo' not in v.ro