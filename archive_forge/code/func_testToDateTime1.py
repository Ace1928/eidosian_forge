import datetime
import pickle
import sys
from copy import deepcopy
from tests.base import BaseTestCase
from pyasn1.type import useful
def testToDateTime1(self):
    assert datetime.datetime(2017, 7, 11, 0, 1, 2, tzinfo=UTC) == useful.UTCTime('170711000102Z').asDateTime