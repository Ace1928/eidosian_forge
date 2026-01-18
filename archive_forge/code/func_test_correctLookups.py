import os
import socket
from unittest import skipIf
from twisted.internet.tcp import ECONNABORTED
from twisted.python.runtime import platform
from twisted.python.win32 import _ErrorFormatter, formatError
from twisted.trial.unittest import TestCase
@skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
def test_correctLookups(self):
    """
        Given a known-good errno, make sure that formatMessage gives results
        matching either C{socket.errorTab}, C{ctypes.WinError}, or
        C{win32api.FormatMessage}.
        """
    acceptable = [socket.errorTab[ECONNABORTED]]
    try:
        from ctypes import WinError
        acceptable.append(WinError(ECONNABORTED).strerror)
    except ImportError:
        pass
    try:
        from win32api import FormatMessage
        acceptable.append(FormatMessage(ECONNABORTED))
    except ImportError:
        pass
    self.assertIn(formatError(ECONNABORTED), acceptable)