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
def getDTPPort(self, factory):
    """
        Return a port for passive access, using C{self.passivePortRange}
        attribute.
        """
    for portn in self.passivePortRange:
        try:
            dtpPort = self.listenFactory(portn, factory)
        except error.CannotListenError:
            continue
        else:
            return dtpPort
    raise error.CannotListenError('', portn, f'No port available in range {self.passivePortRange}')