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
def ftp_NLST(self, path):
    """
        This command causes a directory listing to be sent from the server to
        the client. The pathname should specify a directory or other
        system-specific file group descriptor. An empty path implies the
        current working directory. If the path is non-existent, send nothing.
        If the path is to a file, send only the file name.

        @type path: C{str}
        @param path: The path for which a directory listing should be returned.

        @rtype: L{Deferred}
        @return: a L{Deferred} which will be fired when the listing request
            is finished.
        """
    if self.dtpInstance is None or not self.dtpInstance.isConnected:
        return defer.fail(BadCmdSequenceError('must send PORT or PASV before RETR'))
    try:
        segments = toSegments(self.workingDirectory, path)
    except InvalidPath:
        return defer.fail(FileNotFoundError(path))

    def cbList(results, glob):
        """
            Send, line by line, each matching file in the directory listing,
            and then close the connection.

            @type results: A C{list} of C{tuple}. The first element of each
                C{tuple} is a C{str} and the second element is a C{list}.
            @param results: The names of the files in the directory.

            @param glob: A shell-style glob through which to filter results
                (see U{http://docs.python.org/2/library/fnmatch.html}), or
                L{None} for no filtering.
            @type glob: L{str} or L{None}

            @return: A C{tuple} containing the status code for a successful
                transfer.
            @rtype: C{tuple}
            """
        self.reply(DATA_CNX_ALREADY_OPEN_START_XFR)
        for name, ignored in results:
            if not glob or (glob and fnmatch.fnmatch(name, glob)):
                name = self._encodeName(name)
                self.dtpInstance.sendLine(name)
        self.dtpInstance.transport.loseConnection()
        return (TXFR_COMPLETE_OK,)

    def listErr(results):
        """
            RFC 959 specifies that an NLST request may only return directory
            listings. Thus, send nothing and just close the connection.

            @type results: L{Failure}
            @param results: The L{Failure} wrapping a L{FileNotFoundError} that
                occurred while trying to list the contents of a nonexistent
                directory.

            @returns: A C{tuple} containing the status code for a successful
                transfer.
            @rtype: C{tuple}
            """
        self.dtpInstance.transport.loseConnection()
        return (TXFR_COMPLETE_OK,)
    if _isGlobbingExpression(segments):
        glob = segments.pop()
    else:
        glob = None
    d = self.shell.list(segments)
    d.addCallback(cbList, glob)
    d.addErrback(listErr)
    return d