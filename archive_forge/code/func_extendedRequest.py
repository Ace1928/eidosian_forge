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
def extendedRequest(self, request, data):
    """
        Make an extended request of the server.

        The method returns a Deferred that is called back with
        the result of the extended request.

        @type request: L{bytes}
        @param request: the name of the extended request to make.
        @type data: L{bytes}
        @param data: any other data that goes along with the request.
        """
    return self._sendRequest(FXP_EXTENDED, NS(request) + data)