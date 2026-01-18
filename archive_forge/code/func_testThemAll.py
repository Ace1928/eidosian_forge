from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def testThemAll(self) -> None:
    for entry in self.listOfTests:
        output = entry[0]
        exception = entry[1]
        args = entry[2]
        kwargs = entry[3]
        self.assertEqual(str(exception(*args, **kwargs)), output)