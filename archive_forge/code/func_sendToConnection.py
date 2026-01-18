import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
def sendToConnection(self, commands):
    """
        XXX

        @return: A tuple of two L{Deferred}s:
                  - L{Deferred} L{IFinishableConsumer}. You must call
                    the C{finish} method on the IFinishableConsumer when the
                    file is completely transferred.
                  - L{Deferred} list of control-connection responses.
        """
    s = SenderProtocol()
    r = self._openDataConnection(commands, s)
    return (s.connectedDeferred, r)