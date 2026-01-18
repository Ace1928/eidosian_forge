import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def testPOSIXct_getitem():
    dt = (datetime.datetime(2014, 12, 11) - datetime.datetime(1970, 1, 1)).total_seconds()
    sexp = robjects.r('ISOdate(c(2013, 2014), 12, 11, hour = 0, tz = "UTC")')
    res = robjects.POSIXct(sexp)
    assert res[1] - dt == 0