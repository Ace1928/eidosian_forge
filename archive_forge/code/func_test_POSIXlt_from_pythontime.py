import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def test_POSIXlt_from_pythontime():
    x = [time.struct_time(_dateval_tuple), time.struct_time(_dateval_tuple)]
    res = robjects.POSIXlt(x)
    assert len(x) == 2