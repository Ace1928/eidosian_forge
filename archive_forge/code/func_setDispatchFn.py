from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
def setDispatchFn(self, fn):
    """Set another function to handle elements."""
    self.stream.ElementEvent = fn