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
def registerReference(self, object):
    """
        Store a persistent reference to a local object and map its
        id() to a generated, session-unique ID.

        @param object: a local object
        @return: the generated ID
        """
    assert object is not None
    puid = object.processUniqueID()
    luid = self.luids.get(puid)
    if luid is None:
        if len(self.localObjects) > MAX_BROKER_REFS:
            self.maxBrokerRefsViolations = self.maxBrokerRefsViolations + 1
            if self.maxBrokerRefsViolations > 3:
                self.transport.loseConnection()
                raise Error('Maximum PB reference count exceeded.  Goodbye.')
            raise Error('Maximum PB reference count exceeded.')
        luid = self.newLocalID()
        self.localObjects[luid] = Local(object)
        self.luids[puid] = luid
    else:
        self.localObjects[luid].incref()
    return luid