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
def retrieveFile(self, path, protocol, offset=0):
    """
        Retrieve a file from the given path

        This method issues the 'RETR' FTP command.

        The file is fed into the given Protocol instance.  The data connection
        will be passive if self.passive is set.

        @param path: path to file that you wish to receive.
        @param protocol: a L{Protocol} instance.
        @param offset: offset to start downloading from

        @return: L{Deferred}
        """
    cmds = ['RETR ' + self.escapePath(path)]
    if offset:
        cmds.insert(0, 'REST ' + str(offset))
    return self.receiveFromConnection(cmds, protocol)