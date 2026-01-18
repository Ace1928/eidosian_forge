import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_DoubleQuoteString(self):
    """
        Colorize an integer in double quotes.
        """
    manhole.lastColorizedLine('"1"')