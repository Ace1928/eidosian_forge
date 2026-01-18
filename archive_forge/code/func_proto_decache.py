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
def proto_decache(self, objectID):
    """
        (internal) Decrement the reference count of a cached object.

        If the reference count is zero, free the reference, then send an
        'uncached' directive.

        @param objectID: The object ID.
        """
    refs = self.remotelyCachedObjects[objectID].decref()
    if refs == 0:
        lobj = self.remotelyCachedObjects[objectID]
        cacheable = lobj.object
        perspective = lobj.perspective
        try:
            cacheable.stoppedObserving(perspective, RemoteCacheObserver(self, cacheable, perspective))
        except BaseException:
            log.deferr()
        puid = cacheable.processUniqueID()
        del self.remotelyCachedLUIDs[puid]
        del self.remotelyCachedObjects[objectID]
        self.sendCall(b'uncache', objectID)