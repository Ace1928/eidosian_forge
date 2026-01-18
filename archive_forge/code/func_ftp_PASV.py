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
def ftp_PASV(self):
    """
        Request for a passive connection

        from the rfc::

            This command requests the server-DTP to "listen" on a data port
            (which is not its default data port) and to wait for a connection
            rather than initiate one upon receipt of a transfer command.  The
            response to this command includes the host and port address this
            server is listening on.
        """
    if self.dtpFactory is not None:
        self.cleanupDTP()
    self.dtpFactory = DTPFactory(pi=self)
    self.dtpFactory.setTimeout(self.dtpTimeout)
    self.dtpPort = self.getDTPPort(self.dtpFactory)
    host = self.transport.getHost().host
    port = self.dtpPort.getHost().port
    self.reply(ENTERING_PASV_MODE, encodeHostPort(host, port))
    return self.dtpFactory.deferred.addCallback(lambda ign: None)