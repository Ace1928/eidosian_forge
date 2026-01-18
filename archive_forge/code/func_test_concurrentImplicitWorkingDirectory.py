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
def test_concurrentImplicitWorkingDirectory(self):
    """
        If no working directory is explicitly specified and the default
        working directory is in use by another runner, L{TrialRunner.run}
        selects a different default working directory to use.
        """
    self.parseOptions([])
    self.addCleanup(os.chdir, os.getcwd())
    runDirectory = FilePath(self.mktemp())
    runDirectory.makedirs()
    os.chdir(runDirectory.path)
    firstRunner = self.getRunner()
    secondRunner = self.getRunner()
    where = {}

    class ConcurrentCase(unittest.SynchronousTestCase):

        def test_first(self):
            """
                Start a second test run which will have a default working
                directory which is the same as the working directory of the
                test run already in progress.
                """
            where['concurrent'] = subsequentDirectory = os.getcwd()
            os.chdir(runDirectory.path)
            self.addCleanup(os.chdir, subsequentDirectory)
            secondRunner.run(ConcurrentCase('test_second'))

        def test_second(self):
            """
                Record the working directory for later analysis.
                """
            where['record'] = os.getcwd()
    result = firstRunner.run(ConcurrentCase('test_first'))
    bad = result.errors + result.failures
    if bad:
        self.fail(bad[0][1])
    self.assertEqual(where, {'concurrent': runDirectory.child('_trial_temp').path, 'record': runDirectory.child('_trial_temp-1').path})