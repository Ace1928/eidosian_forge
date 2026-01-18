import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
def test_capturesError(self):
    """
        Chek that a L{LoggedSuite} reports any logged errors to its result.
        """
    result = reporter.TestResult()
    suite = runner.LoggedSuite([BreakingSuite()])
    suite.run(result)
    self.assertEqual(len(result.errors), 1)
    self.assertEqual(result.errors[0][0].id(), runner.NOT_IN_TEST)
    self.assertTrue(result.errors[0][1].check(RuntimeError))