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
def gotServerVersion(self, serverVersion, extData):
    """
        Called when the client sends their version info.

        @param serverVersion: an integer representing the version of the SFTP
        protocol they are claiming.
        @param extData: a dictionary of extended_name : extended_data items.
        These items are sent by the client to indicate additional features.
        """