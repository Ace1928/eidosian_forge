import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def openDirectory(self, path):
    """
        Open a directory for scanning.

        This method returns a Deferred that is called back with an iterable
        object that has a close() method.

        The close() method is called when the client is finished reading
        from the directory.  At this point, the iterable will no longer
        be used.

        The iterable returns triples of the form (filename, longname, attrs)
        or a Deferred that returns the same.  The sequence must support
        __getitem__, but otherwise may be any 'sequence-like' object.

        filename is the name of the file relative to the directory.
        logname is an expanded format of the filename.  The recommended format
        is:
        -rwxr-xr-x   1 mjos     staff      348911 Mar 25 14:29 t-filexfer
        1234567890 123 12345678 12345678 12345678 123456789012

        The first line is sample output, the second is the length of the field.
        The fields are: permissions, link count, user owner, group owner,
        size in bytes, modification time.

        attrs is a dictionary in the format of the attrs argument to openFile.

        @type path: L{bytes}
        @param path: the directory to open.
        """
    d = self._sendRequest(FXP_OPENDIR, NS(path))
    d.addCallback(self._cbOpenHandle, ClientDirectory, path)
    return d