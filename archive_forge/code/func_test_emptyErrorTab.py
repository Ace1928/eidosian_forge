import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
def test_emptyErrorTab(self):
    """
        L{_ErrorFormatter.formatError} should use L{os.strerror} to format
        error messages if it is constructed with only an error tab which does
        not contain the error code it is called with.
        """
    error = 1
    self.assertNotEqual(self.probeErrorCode, error)
    formatter = _ErrorFormatter(None, None, {error: 'wrong message'})
    message = formatter.formatError(self.probeErrorCode)
    self.assertEqual(message, os.strerror(self.probeErrorCode))