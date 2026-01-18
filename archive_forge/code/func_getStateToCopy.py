import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def getStateToCopy(self):
    """Gather state to send when I am serialized for a peer.

        I will default to returning self.__dict__.  Override this to
        customize this behavior.
        """
    return self.__dict__