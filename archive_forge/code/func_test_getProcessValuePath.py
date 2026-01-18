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
def test_getProcessValuePath(self):
    """
        L{getProcessValue} runs the given command with the working directory
        given by the C{path} parameter.
        """

    def check(result, ignored):
        self.assertEqual(result, 0)
    return self._pathTest(utils.getProcessValue, check)