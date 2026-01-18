import logging
import sys
import weakref
import webob
from wsme.exc import ClientSideError, UnknownFunction
from wsme.protocol import getprotocol
from wsme.rest import scan_api
import wsme.api
import wsme.types
def addprotocol(self, protocol, **options):
    """
        Enable a new protocol on the controller.

        :param protocol: A registered protocol name or an instance
                         of a protocol.
        """
    if isinstance(protocol, str):
        protocol = getprotocol(protocol, **options)
    self.protocols.append(protocol)
    protocol.root = weakref.proxy(self)