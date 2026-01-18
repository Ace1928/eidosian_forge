import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def rootObject(self, broker):
    """A factory is requesting to publish me as a root object.

        When a factory is sending me as the root object, this
        method will be invoked to allow per-broker versions of an
        object.  By default I return myself.
        """
    return self