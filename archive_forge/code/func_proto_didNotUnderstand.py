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
def proto_didNotUnderstand(self, command):
    """
        Respond to stock 'C{didNotUnderstand}' message.

        Log the command that was not understood and continue. (Note:
        this will probably be changed to close the connection or raise
        an exception in the future.)

        @param command: The command to log.
        """
    log.msg("Didn't understand command: %r" % command)