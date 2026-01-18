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
def ftp_LIST(self, path=''):
    """This command causes a list to be sent from the server to the
        passive DTP.  If the pathname specifies a directory or other
        group of files, the server should transfer a list of files
        in the specified directory.  If the pathname specifies a
        file then the server should send current information on the
        file.  A null argument implies the user's current working or
        default directory.
        """
    if self.dtpInstance is None or not self.dtpInstance.isConnected:
        return defer.fail(BadCmdSequenceError('must send PORT or PASV before RETR'))
    if path.lower() in ['-a', '-l', '-la', '-al']:
        path = ''

    def gotListing(results):
        self.reply(DATA_CNX_ALREADY_OPEN_START_XFR)
        for name, attrs in results:
            name = self._encodeName(name)
            self.dtpInstance.sendListResponse(name, attrs)
        self.dtpInstance.transport.loseConnection()
        return (TXFR_COMPLETE_OK,)
    try:
        segments = toSegments(self.workingDirectory, path)
    except InvalidPath:
        return defer.fail(FileNotFoundError(path))
    d = self.shell.list(segments, ('size', 'directory', 'permissions', 'hardlinks', 'modified', 'owner', 'group'))
    d.addCallback(gotListing)
    return d