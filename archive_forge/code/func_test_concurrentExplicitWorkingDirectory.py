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
def test_concurrentExplicitWorkingDirectory(self):
    """
        If a working directory which is already in use is explicitly specified,
        L{TrialRunner.run} raises L{_WorkingDirectoryBusy}.
        """
    self.parseOptions(['--temp-directory', os.path.abspath(self.mktemp())])
    initialDirectory = os.getcwd()
    self.addCleanup(os.chdir, initialDirectory)
    firstRunner = self.getRunner()
    secondRunner = self.getRunner()

    class ConcurrentCase(unittest.SynchronousTestCase):

        def test_concurrent(self):
            """
                Try to start another runner in the same working directory and
                assert that it raises L{_WorkingDirectoryBusy}.
                """
            self.assertRaises(util._WorkingDirectoryBusy, secondRunner.run, ConcurrentCase('test_failure'))

        def test_failure(self):
            """
                Should not be called, always fails.
                """
            self.fail('test_failure should never be called.')
    result = firstRunner.run(ConcurrentCase('test_concurrent'))
    bad = result.errors + result.failures
    if bad:
        self.fail(bad[0][1])