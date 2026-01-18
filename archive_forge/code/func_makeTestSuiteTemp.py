from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def makeTestSuiteTemp(classes):
    suite = TestSuite()
    suite.addTest(PyparsingTestInit())
    suite.addTests((cls() for cls in classes))
    return suite