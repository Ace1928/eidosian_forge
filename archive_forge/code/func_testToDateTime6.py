import datetime
import pickle
import sys
from copy import deepcopy
from tests.base import BaseTestCase
from pyasn1.type import useful
def testToDateTime6(self):
    assert datetime.datetime(2017, 7, 11, 0, 1, 2, 30000, tzinfo=UTC2) == useful.GeneralizedTime('20170711000102.3+02').asDateTime