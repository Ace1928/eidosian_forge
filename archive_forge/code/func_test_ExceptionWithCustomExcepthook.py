import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_ExceptionWithCustomExcepthook(self):
    """
        Raised exceptions are handled the same way even if L{sys.excepthook}
        has been modified from its original value.
        """
    self.patch(sys, 'excepthook', lambda *args: None)
    return self.test_Exception()