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
def cleanupDTP(self):
    """
        Call when DTP connection exits
        """
    log.msg('cleanupDTP', debug=True)
    log.msg(self.dtpPort)
    dtpPort, self.dtpPort = (self.dtpPort, None)
    if interfaces.IListeningPort.providedBy(dtpPort):
        dtpPort.stopListening()
    elif interfaces.IConnector.providedBy(dtpPort):
        dtpPort.disconnect()
    else:
        assert False, 'dtpPort should be an IListeningPort or IConnector, instead is %r' % (dtpPort,)
    self.dtpFactory.stopFactory()
    self.dtpFactory = None
    if self.dtpInstance is not None:
        self.dtpInstance = None