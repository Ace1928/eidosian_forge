import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_identicalOutput(self):
    """
        The output of UTF-8 bytestrings and Unicode strings are identical.
        """
    self.assertEqual(manhole.lastColorizedLine(b'\xd0\xb8'), manhole.lastColorizedLine('и'))