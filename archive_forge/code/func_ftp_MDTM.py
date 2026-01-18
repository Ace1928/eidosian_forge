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
def ftp_MDTM(self, path):
    """
        File Modification Time (MDTM)

        The FTP command, MODIFICATION TIME (MDTM), can be used to determine
        when a file in the server NVFS was last modified.  This command has
        existed in many FTP servers for many years, as an adjunct to the REST
        command for STREAM mode, thus is widely available.  However, where
        supported, the "modify" fact that can be provided in the result from
        the new MLST command is recommended as a superior alternative.

        http://tools.ietf.org/html/rfc3659
        """
    try:
        newsegs = toSegments(self.workingDirectory, path)
    except InvalidPath:
        return defer.fail(FileNotFoundError(path))

    def cbStat(result):
        modified, = result
        return (FILE_STATUS, time.strftime('%Y%m%d%H%M%S', time.gmtime(modified)))
    return self.shell.stat(newsegs, ('modified',)).addCallback(cbStat)