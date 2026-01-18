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
def cachedRemotelyAs(self, instance, incref=0):
    """

        @param instance: The instance to look up.
        @param incref: Flag to specify whether to increment the
                       reference.
        @return: An ID that says what this instance is cached as
                 remotely, or L{None} if it's not.
        """
    puid = instance.processUniqueID()
    luid = self.remotelyCachedLUIDs.get(puid)
    if luid is not None and incref:
        self.remotelyCachedObjects[luid].incref()
    return luid