import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
@skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
def test_fromEnvironment(self):
    """
        L{_ErrorFormatter.fromEnvironment} should create an L{_ErrorFormatter}
        instance with attributes populated from available modules.
        """
    formatter = _ErrorFormatter.fromEnvironment()
    if formatter.winError is not None:
        from ctypes import WinError
        self.assertEqual(formatter.formatError(self.probeErrorCode), WinError(self.probeErrorCode).strerror)
        formatter.winError = None
    if formatter.formatMessage is not None:
        from win32api import FormatMessage
        self.assertEqual(formatter.formatError(self.probeErrorCode), FormatMessage(self.probeErrorCode))
        formatter.formatMessage = None
    if formatter.errorTab is not None:
        from socket import errorTab
        self.assertEqual(formatter.formatError(self.probeErrorCode), errorTab[self.probeErrorCode])