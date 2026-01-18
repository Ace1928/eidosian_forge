import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def testPOSIXct_fromSexp():
    sexp = robjects.r('ISOdate(2013, 12, 11)')
    res = robjects.POSIXct(sexp)
    assert len(res) == 1