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