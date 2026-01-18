from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
def resetDispatchFn(self):
    """Set the default function (C{onElement}) to handle elements."""
    self.stream.ElementEvent = self.onElement