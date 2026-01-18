import os
import signal
import stat
import sys
import warnings
from unittest import skipIf
from twisted.internet import error, interfaces, reactor, utils
from twisted.internet.defer import Deferred
from twisted.python.runtime import platform
from twisted.python.test.test_util import SuppressedWarningsTests
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_getProcessOutputDefaultPath(self):
    """
        If no value is supplied for the C{path} parameter, L{getProcessOutput}
        runs the given command in the same working directory as the parent
        process and succeeds even if the current working directory is not
        accessible.
        """
    return self._defaultPathTest(utils.getProcessOutput, self.assertEqual)