import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def gotPartialLine(ign):
    self._assertBuffer([b'>>> cancelled line'])
    self._testwrite(manhole.CTRL_BACKSLASH)
    d = self.recvlineClient.onDisconnection
    return self.assertFailure(d, error.ConnectionDone)