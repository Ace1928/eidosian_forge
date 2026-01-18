import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def gotInterruption(ignored):
    self._assertBuffer([b'>>> (', b'... things', b'KeyboardInterrupt', b'>>> '])