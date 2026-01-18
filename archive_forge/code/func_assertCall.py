import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def assertCall(self, occurrence, methodName, expectedPositionalArgs=(), expectedKeywordArgs={}):
    attr, mock = occurrence
    self.assertEqual(attr, methodName)
    self.assertEqual(len(occurrences(mock)), 1)
    [(call, result, args, kw)] = occurrences(mock)
    self.assertEqual(call, '__call__')
    self.assertEqual(args, expectedPositionalArgs)
    self.assertEqual(kw, expectedKeywordArgs)
    return result