import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
def localObjectForID(self, luid):
    """
        Get a local object for a locally unique ID.

        @return: An object previously stored with L{registerReference} or
            L{None} if there is no object which corresponds to the given
            identifier.
        """
    if isinstance(luid, str):
        luid = luid.encode('utf8')
    lob = self.localObjects.get(luid)
    if lob is None:
        return
    return lob.object