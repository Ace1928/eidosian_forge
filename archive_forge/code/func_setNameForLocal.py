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
def setNameForLocal(self, name, object):
    """
        Store a special (string) ID for this object.

        This is how you specify a 'base' set of objects that the remote
        protocol can connect to.

        @param name: An ID.
        @param object: The object.
        """
    if isinstance(name, str):
        name = name.encode('utf8')
    assert object is not None
    self.localObjects[name] = Local(object)