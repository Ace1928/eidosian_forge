import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def deliverResponse(self, responseMessage):
    """
        Deliver response.

        Destination is based on topmost Via header.
        """
    destVia = parseViaHeader(responseMessage.headers['via'][0])
    host = destVia.received or destVia.host
    port = destVia.rport or destVia.port or self.PORT
    destAddr = URL(host=host, port=port)
    self.sendMessage(destAddr, responseMessage)