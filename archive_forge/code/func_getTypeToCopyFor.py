import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def getTypeToCopyFor(self, perspective):
    """Determine what type tag to send for me.

        By default, defer to self.L{getTypeToCopy}() normally this is
        adequate, but you may override this to change it.
        """
    return self.getTypeToCopy()