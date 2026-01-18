import fcntl
import fnmatch
import getpass
import glob
import os
import pwd
import stat
import struct
import sys
import tty
from typing import List, Optional, TextIO, Union
from twisted.conch.client import connect, default, options
from twisted.conch.ssh import channel, common, connection, filetransfer
from twisted.internet import defer, reactor, stdio, utils
from twisted.protocols import basic
from twisted.python import failure, log, usage
from twisted.python.filepath import FilePath
def cmd_PUT(self, rest):
    """
        Do an upload request for a single local file or a globing expression.

        @param rest: Requested command line for the PUT command.
        @type rest: L{str}

        @return: A deferred which fires with L{None} when transfer is done.
        @rtype: L{defer.Deferred}
        """
    local, rest = self._getFilename(rest)
    if '*' in local or '?' in local:
        if rest:
            remote, rest = self._getFilename(rest)
            remote = os.path.join(self.currentDirectory, remote)
        else:
            remote = ''
        files = glob.glob(local)
        return self._putMultipleFiles(files, remote)
    else:
        if rest:
            remote, rest = self._getFilename(rest)
        else:
            remote = os.path.split(local)[1]
        return self._putSingleFile(local, remote)