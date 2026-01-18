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
def remoteForName(self, name):
    """
        Returns an object from the remote name mapping.

        Note that this does not check the validity of the name, only
        creates a translucent reference for it.

        @param name: The name to look up.
        @return: An object which maps to the name.
        """
    if isinstance(name, str):
        name = name.encode('utf8')
    return RemoteReference(None, self, name, 0)