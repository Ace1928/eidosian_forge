from __future__ import annotations
import errno
import os
import signal
from twisted.python.runtime import platformType
from twisted.trial.unittest import SynchronousTestCase
def test_isDefaultHandler(self) -> None:
    """
        L{isDefaultHandler} returns true if the SIGCHLD handler is SIG_DFL,
        false otherwise.
        """
    self.assertTrue(isDefaultHandler())
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    self.assertFalse(isDefaultHandler())
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    self.assertTrue(isDefaultHandler())
    signal.signal(signal.SIGCHLD, lambda *args: None)
    self.assertFalse(isDefaultHandler())