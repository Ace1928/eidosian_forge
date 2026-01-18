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
def test_getProcessOutputPath(self):
    """
        L{getProcessOutput} runs the given command with the working directory
        given by the C{path} parameter.
        """
    return self._pathTest(utils.getProcessOutput, self.assertEqual)