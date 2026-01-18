import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def remoteMethod(self, key):
    """Get a L{pb.RemoteMethod} for this key."""
    return RemoteCacheMethod(key, self.broker, self.cached, self.perspective)