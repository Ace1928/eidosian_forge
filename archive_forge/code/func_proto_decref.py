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
def proto_decref(self, objectID):
    """
        (internal) Decrement the reference count of an object.

        If the reference count is zero, it will free the reference to this
        object.

        @param objectID: The object ID.
        """
    if isinstance(objectID, str):
        objectID = objectID.encode('utf8')
    refs = self.localObjects[objectID].decref()
    if refs == 0:
        puid = self.localObjects[objectID].object.processUniqueID()
        del self.luids[puid]
        del self.localObjects[objectID]
        self._localCleanup.pop(puid, lambda: None)()