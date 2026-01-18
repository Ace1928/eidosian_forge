from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def makeTestSuite():
    import inspect
    suite = TestSuite()
    suite.addTest(PyparsingTestInit())
    test_case_classes = ParseTestCase.__subclasses__()
    test_case_classes.sort(key=lambda cls: inspect.getsourcelines(cls)[1])
    test_case_classes.remove(PyparsingTestInit)
    test_case_classes.remove(EnablePackratParsing)
    if IRON_PYTHON_ENV:
        test_case_classes.remove(OriginalTextForTest)
    suite.addTests((T() for T in test_case_classes))
    if TEST_USING_PACKRAT:
        suite.addTest(EnablePackratParsing())
        unpackrattables = [PyparsingTestInit, EnablePackratParsing, RepeaterTest]
        packratTests = [t.__class__() for t in suite._tests if t.__class__ not in unpackrattables]
        suite.addTests(packratTests)
    return suite